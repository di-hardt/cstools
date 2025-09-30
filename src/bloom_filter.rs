use std::{f64::consts::E, io::Read};

use anyhow::{bail, Result};
use bitvec::prelude::*;
use murmur3::murmur3_x64_128 as murmur3hash;

/// BloomFilter using murmur3 hash functions
///
#[derive(Clone)]
pub struct BloomFilter<T>
where
    T: BitStore,
{
    /// False positive probability
    pub(crate) fp_prob: f64,

    /// Length of the bloom filter as u128 fosr position calculation
    pub(crate) length: u128,

    // Number of hash functions to apply
    pub(crate) hash_count: u32,

    // Bit vector
    pub(crate) bitvec: BitBox<T, Msb0>,
}

impl<T> BloomFilter<T>
where
    T: BitStore,
{
    /// Creates a new Bloom filter
    ///
    /// Arguments:
    /// * `fp_prob` - False Positive probability in decimal
    /// * `length` - Length of the bloom filter
    /// * `hash_count` - Number of hash functions to use
    /// * `bitvec` - Bit vector
    ///
    pub(crate) fn new(fp_prob: f64, hash_count: u32, bitvec: BitBox<T, Msb0>) -> Result<Self> {
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
    pub fn bitvec(&self) -> &BitBox<T, Msb0> {
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
        let bitvec = bitvec!(T, Msb0; 0; length as usize);

        Self::new(fp_prob, hash_count, bitvec.into_boxed_bitslice())
    }

    /// Creates a bloom filter with the given size and false positive probability
    ///
    /// # Arguments
    /// * `length` - Length of the bloom filter
    /// * `fp_prob` - False Positive probability in decimal
    ///
    pub fn new_by_length_and_fp_prob(length: u64, fp_prob: f64) -> Result<Self> {
        let rounded_length = Self::round_to_t(length);

        let (_, hash_count) = Self::calc_item_size_and_hash_count(rounded_length, fp_prob);

        // Bit array of given size
        let bitvec = bitvec!(T, Msb0; 0; rounded_length as usize);

        Self::new(fp_prob, hash_count, bitvec.into_boxed_bitslice())
    }

    /// Calculates the strings position within the bitvecotor
    ///
    /// # Arguments
    /// * `item` - Item to calculate position for
    /// * `seed` - Seed to use for murmur3 hash
    ///
    fn calc_item_position<I>(&self, item: &mut I, seed: u32) -> Result<usize>
    where
        I: Read,
    {
        Ok((murmur3hash(item, seed)? % self.length) as usize)
    }

    /// Add an item to the filter
    ///
    /// # Arguments
    ///
    /// * `item` - Item to add
    ///
    pub fn add<I>(&mut self, item: &mut I) -> Result<()>
    where
        I: Read,
    {
        for i in 0..self.hash_count {
            // Create hash for given item.
            // `i` works as seed to mmh3.hash() function
            let digest = self.calc_item_position(item, i)?;
            // Set the bit to true
            self.bitvec.set(digest, true)
        }
        Ok(())
    }

    /// Check for existence of the given ixtem in filter
    ///
    /// # Arguments
    /// * `item` - Item to search
    ///
    pub fn contains<I>(&self, item: &mut I) -> Result<bool>
    where
        I: Read,
    {
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
    /// Rounded up to nearest multiple of T
    ///
    /// # Arguments
    ///
    /// `n` - Number of items expected to be stored in filter
    /// `p` - False Positive probability in decimal
    ///
    pub fn calc_length(n: u64, p: f64) -> u64 {
        let m = (-(n as f64 * p.log(E)) / (2.0_f64.log(E).powi(2))) as u64;
        Self::round_to_t(m)
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

    pub fn round_to_t(length: u64) -> u64 {
        let remains = length % (std::mem::size_of::<T>() * 8) as u64;
        if remains == 0 {
            length
        } else {
            length + (std::mem::size_of::<T>() * 8) as u64 - remains
        }
    }
}

impl<T> BloomFilter<T>
where
    T: BitStore + radium::Radium,
{
    /// Add an item to the filter
    ///
    /// This is equivalent to [`.add()`], except that it does not require an
    /// `&mut` reference.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to add
    ///
    pub fn add_aliased<I>(&self, item: &mut I) -> Result<()>
    where
        I: Read,
    {
        for i in 0..self.hash_count {
            // Create hash for given item.
            // `i` works as seed to mmh3.hash() function
            let digest = self.calc_item_position(item, i)?;
            // Set the bit to true
            self.bitvec.set_aliased(digest, true)
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::fs::read_to_string;
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, AtomicUsize};

    use super::*;

    /// Inerting and finding with mutable reference
    ///
    #[test]
    fn test_inserting_and_finding_mut() {
        test_inserting_and_finding_mut_generic::<u8>();
        test_inserting_and_finding_mut_generic::<u16>();
        test_inserting_and_finding_mut_generic::<u32>();
        test_inserting_and_finding_mut_generic::<u64>();
        test_inserting_and_finding_mut_generic::<usize>();

        test_inserting_and_finding_mut_generic::<Cell<u8>>();
        test_inserting_and_finding_mut_generic::<Cell<u16>>();
        test_inserting_and_finding_mut_generic::<Cell<u32>>();
        test_inserting_and_finding_mut_generic::<Cell<u64>>();
        test_inserting_and_finding_mut_generic::<Cell<usize>>();

        test_inserting_and_finding_mut_generic::<AtomicU8>();
        test_inserting_and_finding_mut_generic::<AtomicU16>();
        test_inserting_and_finding_mut_generic::<AtomicU32>();
        test_inserting_and_finding_mut_generic::<AtomicU64>();
        test_inserting_and_finding_mut_generic::<AtomicUsize>();
    }

    fn test_inserting_and_finding_mut_generic<T>()
    where
        T: BitStore,
    {
        let some_strings: Vec<String> =
            read_to_string(PathBuf::from("test_data/10000_random_strings.txt"))
                .unwrap()
                .lines()
                .map(String::from)
                .collect();

        let mut bloom_filter: BloomFilter<T> =
            BloomFilter::new_by_item_count_and_fp_prob(some_strings.len() as u64, 0.01).unwrap();

        for a_string in some_strings.iter() {
            bloom_filter
                .add(&mut Cursor::new(a_string.as_bytes()))
                .unwrap();
        }

        for a_string in some_strings.iter() {
            assert!(bloom_filter
                .contains(&mut Cursor::new(a_string.as_bytes()))
                .unwrap());
        }
    }

    /// Inerting and finding with using atomic operations
    ///
    #[test]
    fn test_inserting_and_finding() {
        test_inserting_and_finding_generic::<Cell<u8>>();
        test_inserting_and_finding_generic::<Cell<u16>>();
        test_inserting_and_finding_generic::<Cell<u32>>();
        test_inserting_and_finding_generic::<Cell<u64>>();
        test_inserting_and_finding_generic::<Cell<usize>>();

        test_inserting_and_finding_generic::<Cell<usize>>();
        test_inserting_and_finding_generic::<AtomicU8>();
        test_inserting_and_finding_generic::<AtomicU16>();
        test_inserting_and_finding_generic::<AtomicU32>();
        test_inserting_and_finding_generic::<AtomicU64>();
        test_inserting_and_finding_generic::<AtomicUsize>();
    }

    fn test_inserting_and_finding_generic<T>()
    where
        T: BitStore + radium::Radium,
    {
        let some_strings: Vec<String> =
            read_to_string(PathBuf::from("test_data/10000_random_strings.txt"))
                .unwrap()
                .lines()
                .map(String::from)
                .collect();

        let bloom_filter: BloomFilter<T> =
            BloomFilter::new_by_item_count_and_fp_prob(some_strings.len() as u64, 0.01).unwrap();

        for a_string in some_strings.iter() {
            bloom_filter
                .add_aliased(&mut Cursor::new(a_string.as_bytes()))
                .unwrap();
        }

        for a_string in some_strings.iter() {
            assert!(bloom_filter
                .contains(&mut Cursor::new(a_string.as_bytes()))
                .unwrap());
        }
    }
}
