use std::io::Cursor;
use std::{f64::consts::E, sync::atomic::AtomicU8};

use anyhow::{bail, Result};
use bitvec::prelude::*;
use murmur3::murmur3_x64_128 as murmur3hash;

/// Thread safe BloomFilter struct for saving strings. Using murmur3 hash function
///
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone)]
pub struct BloomFilter {
    /// False positive probability
    fp_prob: f64,

    /// Length of the bloom filter as u128 for position calculation
    #[cfg_attr(feature = "serde", serde(skip))]
    length: u128,

    // Number of hash functions to apply
    hash_count: u32,

    // Bit vector
    bitvec: BitBox<AtomicU8, Msb0>,
}

impl BloomFilter {
    /// Creates a new Bloom filter
    ///
    /// Arguments:
    /// * `fp_prob` - False Positive probability in decimal
    /// * `length` - Length of the bloom filter
    /// * `hash_count` - Number of hash functions to use
    /// * `bitvec` - Bit vector
    ///
    fn new(fp_prob: f64, hash_count: u32, bitvec: BitBox<AtomicU8, Msb0>) -> Result<Self> {
        let length = bitvec.len() as u128;

        Ok(Self {
            fp_prob,
            hash_count,
            length,
            bitvec,
        })
    }

    /// Get false positive probability
    ///
    pub fn get_fp_prob(&self) -> f64 {
        self.fp_prob
    }

    /// Size of bit vec in bytes
    ///
    pub fn get_size(&self) -> usize {
        self.bitvec.len() / 8
    }

    /// Get number of hash functions
    ///
    pub fn hash_count(&self) -> u32 {
        self.hash_count
    }

    /// Get bit vector
    ///
    pub fn bitvec(&self) -> &BitBox<AtomicU8, Msb0> {
        &self.bitvec
    }

    /// Length of bit vector
    ///
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.bitvec.len()
    }

    /// Creates new bloom filter for the given number of items and false positive probability
    ///
    /// # Arguments
    /// * `number_of_item` - Number of items expected to be stored in the bloom filter
    /// * `fp_prob` - False Positive probability in decimal
    ///
    pub fn new_by_item_count_and_fp_prob(number_of_item: u64, fp_prob: f64) -> Result<Self> {
        // length of bit array to use
        let length = Self::calc_length(number_of_item, fp_prob);

        // Number of hash functions to use
        let hash_count = Self::calc_hash_count(length, number_of_item)?;

        // Bit array of given size
        let bitvec = bitvec!(AtomicU8, Msb0; 0; length as usize);

        Self::new(fp_prob, hash_count, bitvec.into_boxed_bitslice())
    }

    /// Creates a bloom filter with the given size and false positive probability
    ///
    /// # Arguments
    /// * `length` - Length of the bloom filter
    /// * `fp_prob` - False Positive probability in decimal
    ///
    pub fn new_by_length_and_fp_prob(length: u64, fp_prob: f64) -> Result<Self> {
        let rounded_length = length + 8 - (length % 8);

        let (_, hash_count) = Self::calc_item_size_and_hash_count(rounded_length, fp_prob);

        // Bit array of given size
        let bitvec = bitvec!(AtomicU8, Msb0; 0; rounded_length as usize);

        Self::new(fp_prob, hash_count, bitvec.into_boxed_bitslice())
    }

    /// Calculates the strings position within the bitvecotor
    ///
    /// # Arguments
    /// * `item` - Item to calculate position for
    /// * `seed` - Seed to use for murmur3 hash
    ///
    fn calc_item_position(&self, item: &str, seed: u32) -> Result<usize> {
        Ok((murmur3hash(&mut Cursor::new(item), seed)? % self.length) as usize)
    }

    /// Add an item in the filter
    ///
    /// # Arguments
    ///
    /// * `item` - Item to add
    ///
    pub fn add(&mut self, item: &str) -> Result<()> {
        for i in 0..self.hash_count {
            // Create hash for given item.
            // `i` works as seed to mmh3.hash() function
            let digest = self.calc_item_position(item, i)?;
            // Set the bit to true
            self.bitvec.set(digest, true)
        }
        Ok(())
    }

    /// Add an item to the filter
    ///
    /// This is equivalent to [`.add()`], except that it does not require an
    /// `&mut` reference.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to add
    ///
    pub fn add_aliased(&self, item: &str) -> Result<()> {
        for i in 0..self.hash_count {
            // Create hash for given item.
            // `i` works as seed to mmh3.hash() function
            let digest = self.calc_item_position(item, i)?;
            // Set the bit to true
            self.bitvec.set_aliased(digest, true)
        }
        Ok(())
    }

    /// Check for existence of the given item in filter
    ///
    /// # Arguments
    /// * `item` - Item to search
    ///
    pub fn contains(&self, item: &str) -> Result<bool> {
        for i in 0..self.hash_count {
            let digest = self.calc_item_position(item, i)?;
            if !self.bitvec[digest] {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Calculates the length of the bit array `m` using
    /// the following formula
    /// m = -(n * lg(p)) / (lg(2)^2)
    ///
    /// Rounded up to nearest multiple of 8
    ///
    /// # Arguments
    ///
    /// `n` - Number of items expected to be stored in filter
    /// `p` - False Positive probability in decimal
    ///
    pub fn calc_length(n: u64, p: f64) -> u64 {
        let mut m = (-(n as f64 * p.log(E)) / (2.0_f64.log(E).powi(2))) as u64;
        m += 8 - (m % 8); // round up to nearest multiple of 8
        m
    }

    /// Calculates the number of hash function `k` to apply when checking for an item, using
    /// following formula
    /// k = (m/n) * lg(2)
    ///
    /// # Arguments
    ///
    /// * `m` - Length of bit array
    /// * `n` - Number of items expected to be stored in filter
    ///
    pub fn calc_hash_count(m: u64, n: u64) -> Result<u32> {
        let k = ((m as f64) / (n as f64)) * 2.0_f64.log(E);
        if k > u32::MAX as f64 {
            bail!("Hash count is too large");
        }
        Ok(k as u32)
    }

    /// Calculates item size and hash count
    /// by increasing the hash_count to fit the maximum possible number of items.
    ///
    /// # Arguments
    /// * `hash_count` - Number of hash functions to use
    /// * `fp_prob` - False Positive probability in decimal
    ///
    pub fn calc_item_size_and_hash_count(size: u64, fp_prob: f64) -> (u64, u32) {
        let size_f = size as f64;
        let mut item_size: u64 = 0;
        for i in 1..=u32::MAX {
            let i_f = i as f64;
            let temp_item_size =
                (size_f / (-i_f / (1_f64 - (fp_prob.ln() / i_f).exp()).ln())).ceil() as u64;
            if item_size > temp_item_size {
                return (item_size, i - 1);
            } else {
                item_size = temp_item_size;
            }
        }
        (item_size, u32::MAX)
    }

    /// Loads bloom filter from hdf5 file
    ///
    /// # Arguments
    /// * `path` - Path to hdf5 file
    ///
    #[cfg(feature = "hdf5")]
    pub fn load_hdf5(path: &std::path::PathBuf) -> Result<Self> {
        let file = hdf5::File::open(path)?;
        let hash_count = file.dataset("hash_count")?.read_scalar::<u32>()?;
        let fp_prob = file.dataset("fp_prob")?.read_scalar::<f64>()?;
        let bytes = match Self::decode_hex(
            file.dataset("bit_array")?
                .read_scalar::<hdf5::types::VarLenAscii>()?
                .as_str(),
        ) {
            Ok(bytes) => bytes,
            Err(err) => bail!(format!("Error while decoding hex: {}", err)),
        };
        Self::new(
            fp_prob,
            hash_count,
            BitVec::<AtomicU8, Msb0>::from_slice(&bytes).into_boxed_bitslice(),
        )
    }

    /// Saves bloom filter to hdf5 file
    ///
    /// # Arguments
    /// * `path` - Path to hdf5 file
    ///
    #[cfg(feature = "hdf5")]
    pub fn save_hdf5(&self, path: &std::path::PathBuf) -> Result<()> {
        let file = hdf5::File::create(path)?;
        file.new_dataset::<u32>()
            .create("hash_count")?
            .write_scalar(&self.hash_count)?;
        file.new_dataset::<f64>()
            .create("fp_prob")?
            .write_scalar(&self.fp_prob)?;
        // Convert bitvec to hex string
        let s_ascii = Self::encode_hex(&self.bitvec)?
            .iter()
            .map(|b| format!("{:02X}", b))
            .collect::<String>();
        // Save hex string to hdf5 file
        file.new_dataset::<hdf5::types::VarLenAscii>()
            .create("bit_array")?
            .write_scalar(&hdf5::types::VarLenAscii::from_ascii(&s_ascii)?)?;
        Ok(())
    }

    /// Decodes hex string to bytes
    ///
    /// # Arguments
    /// * `s` - Hex string
    ///
    #[cfg(feature = "hdf5")]
    pub fn decode_hex(s: &str) -> Result<Vec<AtomicU8>, core::num::ParseIntError> {
        (0..s.len())
            .step_by(2)
            .map(|i| match u8::from_str_radix(&s[i..i + 2], 16) {
                Ok(b) => Ok(AtomicU8::new(b)),
                Err(err) => Err(err),
            })
            .collect()
    }

    /// Encodes bytes to hex string
    ///
    /// # Arguments
    /// * `bit_array` - Bit array
    ///
    #[cfg(feature = "hdf5")]
    pub fn encode_hex(bit_array: &BitBox<AtomicU8, Msb0>) -> Result<Vec<u8>> {
        let mut bytes: Vec<u8> = Vec::with_capacity(bit_array.len() / 8);
        for start in (0..bit_array.len()).step_by(8) {
            bytes.push(bit_array[start..(start + 8)].load::<u8>());
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::read_to_string;
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_inserting_and_finding() {
        let some_strings: Vec<String> =
            read_to_string(PathBuf::from("test_data/10000_random_strings.txt"))
                .unwrap()
                .lines()
                .map(String::from)
                .collect();

        let mut bloom_filter =
            BloomFilter::new_by_item_count_and_fp_prob(some_strings.len() as u64, 0.01).unwrap();

        let some_strings_split = some_strings.split_at(some_strings.len() / 2);

        for a_string in some_strings_split.0.iter() {
            bloom_filter.add(a_string).unwrap();
        }

        for a_string in some_strings_split.1.iter() {
            bloom_filter.add_aliased(a_string).unwrap();
        }

        for a_string in some_strings.iter() {
            assert!(bloom_filter.contains(a_string).unwrap());
        }
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn test_save_and_load() {
        let some_strings: Vec<String> =
            read_to_string(PathBuf::from("test_data/10000_random_strings.txt"))
                .unwrap()
                .lines()
                .map(String::from)
                .collect();

        let mut bloom_filter =
            BloomFilter::new_by_item_count_and_fp_prob(some_strings.len() as u64, 0.01).unwrap();

        for a_string in some_strings.iter() {
            bloom_filter.add(a_string).unwrap();
        }

        let temp_file = std::env::temp_dir().join("bloom_filter.h5");
        if temp_file.is_file() {
            std::fs::remove_file(&temp_file).unwrap();
        }

        bloom_filter.save_hdf5(&temp_file).unwrap();

        let read_bloom_filter = BloomFilter::load_hdf5(&temp_file).unwrap();

        assert!(bloom_filter.len() == read_bloom_filter.len());
        assert!(bloom_filter.hash_count == read_bloom_filter.hash_count);
        assert!(bloom_filter.fp_prob == read_bloom_filter.fp_prob);
        assert!(bloom_filter.bitvec == read_bloom_filter.bitvec);

        for a_string in some_strings.iter() {
            assert!(read_bloom_filter.contains(a_string).unwrap());
        }

        if temp_file.is_file() {
            std::fs::remove_file(&temp_file).unwrap();
        }
    }

    /// Using Serde to serialize and deserialize the bloom filter
    ///
    #[cfg(feature = "serde")]
    #[test]
    fn test_serde() {
        use rmp_serde::{Deserializer, Serializer};
        use serde::{Deserialize, Serialize};

        let some_strings: Vec<String> =
            read_to_string(PathBuf::from("test_data/10000_random_strings.txt"))
                .unwrap()
                .lines()
                .map(String::from)
                .collect();

        let mut bloom_filter =
            BloomFilter::new_by_item_count_and_fp_prob(some_strings.len() as u64, 0.01).unwrap();

        for a_string in some_strings.iter() {
            bloom_filter.add(a_string).unwrap();
        }

        let temp_file_path = std::env::temp_dir().join("bloom_filter.messagepack");
        if temp_file_path.is_file() {
            std::fs::remove_file(&temp_file_path).unwrap();
        }

        let mut temp_file = std::fs::File::create(&temp_file_path).unwrap();
        let mut byte_writer = std::io::BufWriter::new(&mut temp_file);

        bloom_filter
            .serialize(&mut Serializer::new(&mut byte_writer))
            .unwrap();

        drop(byte_writer);

        let mut temp_file = std::fs::File::open(&temp_file_path).unwrap();

        let mut byte_reader = std::io::BufReader::new(&mut temp_file);
        let read_bloom_filter =
            BloomFilter::deserialize(&mut Deserializer::new(&mut byte_reader)).unwrap();

        assert!(bloom_filter.len() == read_bloom_filter.len());
        assert!(bloom_filter.hash_count == read_bloom_filter.hash_count);
        assert!(bloom_filter.fp_prob == read_bloom_filter.fp_prob);
        assert!(bloom_filter.bitvec == read_bloom_filter.bitvec);

        for a_string in some_strings.iter() {
            assert!(read_bloom_filter.contains(a_string).unwrap());
        }

        if temp_file_path.is_file() {
            std::fs::remove_file(&temp_file_path).unwrap();
        }
    }
}
