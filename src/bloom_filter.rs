/// Error handling
pub mod error;

/// Implementation for HDF5 serialization and deserialization
#[cfg(feature = "hdf5")]
pub mod hdf5_utils;

/// Implementation for Serde serialization and deserialization
#[cfg(feature = "serde")]
pub mod serde_utils;

use std::{f64::consts::E, fmt::Display, io::Read};

use bitvec::prelude::*;
use murmur3::murmur3_x64_128 as murmur3hash;

use crate::bloom_filter::error::BloomFilterError;

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

    /// Number of items the filter is designed to hold
    pub(crate) number_of_items: u64,

    /// Number of hash functions to apply
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
    pub(crate) fn new(
        fp_prob: f64,
        hash_count: u32,
        number_of_items: u64,
        bitvec: BitBox<T, Msb0>,
    ) -> Self {
        let length = bitvec.len() as u128;

        Self {
            fp_prob,
            hash_count,
            number_of_items,
            length,
            bitvec,
        }
    }

    /// Get false positive probability
    ///
    pub fn fp_prob(&self) -> f64 {
        self.fp_prob
    }

    /// Number of items the filter is designed to hold
    ///
    pub fn number_of_items(&self) -> u64 {
        self.number_of_items
    }

    /// Size of bit vec in bytes
    ///
    pub fn size(&self) -> usize {
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
    pub fn new_by_item_count_and_fp_prob(
        number_of_item: u64,
        fp_prob: f64,
    ) -> Result<Self, BloomFilterError> {
        // length of bit array to use
        let length = Self::calc_length_rounded(number_of_item, fp_prob);

        if length == 0 {
            return Err(BloomFilterError::LengthZero);
        }

        // Number of hash functions to use
        let hash_count = Self::calc_hash_count(length, number_of_item)?;

        // Bit array of given size
        let bitvec = bitvec!(T, Msb0; 0; length as usize);

        Ok(Self::new(
            fp_prob,
            hash_count,
            number_of_item,
            bitvec.into_boxed_bitslice(),
        ))
    }

    /// Creates a bloom filter with the given size and false positive probability
    ///
    /// # Arguments
    /// * `length` - Length of the bloom filter in bits
    /// * `fp_prob` - False Positive probability in decimal
    ///
    pub fn new_by_length_and_fp_prob(length: u64, fp_prob: f64) -> Result<Self, BloomFilterError> {
        let rounded_length = Self::round_to_t(length);

        if rounded_length == 0 {
            return Err(BloomFilterError::LengthZero);
        }

        let (number_of_items, hash_count) =
            Self::calc_item_size_and_hash_count(rounded_length, fp_prob);

        // Bit array of given size
        let bitvec = bitvec!(T, Msb0; 0; rounded_length as usize);

        Ok(Self::new(
            fp_prob,
            hash_count,
            number_of_items,
            bitvec.into_boxed_bitslice(),
        ))
    }

    /// Calculates the strings position within the bitvecotor
    ///
    /// # Arguments
    /// * `item` - Item to calculate position for
    /// * `seed` - Seed to use for murmur3 hash
    ///
    fn calc_item_position<I>(&self, item: &mut I, seed: u32) -> Result<usize, BloomFilterError>
    where
        I: Read,
    {
        let hash = murmur3hash(item, seed).map_err(BloomFilterError::Hashing)?;
        Ok((hash % self.length) as usize)
    }

    /// Add an item to the filter
    ///
    /// # Arguments
    ///
    /// * `item` - Item to add
    ///
    pub fn add<I>(&mut self, item: &mut I) -> Result<(), BloomFilterError>
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
    pub fn contains<I>(&self, item: &mut I) -> Result<bool, BloomFilterError>
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
    /// m = ceil((n * log(p)) / log(1 / pow(2, log(2))));
    ///
    /// # Arguments
    ///
    /// `n` - Number of items expected to be stored in filter
    /// `p` - False Positive probability in decimal
    ///
    pub fn calc_length(n: u64, p: f64) -> u64 {
        ((n as f64 * p.log(E)) // (n * log(p))
            /
            (1.0_f64 / 2.0_f64.powf(2.0_f64.log(E))).log(E))
        .ceil() as u64
    }

    /// Calculates the length of the bit array `m` using
    /// the following formula
    /// m = ceil((n * log(p)) / log(1 / pow(2, log(2))));
    ///
    /// Rounded up to nearest multiple of T
    ///
    /// # Arguments
    ///
    /// `n` - Number of items expected to be stored in filter
    /// `p` - False Positive probability in decimal
    ///
    pub fn calc_length_rounded(n: u64, p: f64) -> u64 {
        Self::round_to_t(Self::calc_length(n, p))
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
    pub fn calc_hash_count(m: u64, n: u64) -> Result<u32, BloomFilterError> {
        let k = (((m as f64) / (n as f64)) * 2.0_f64.log(E)).round();
        if k > u32::MAX as f64 {
            return Err(BloomFilterError::HashCountTooLarge);
        }
        Ok(k as u32)
    }

    /// Calculates the maximum number of items `n` the filter can hold
    ///
    /// ceil(m / (-k / log(1 - exp(log(p) / k))))
    ///
    /// # Arguments
    /// * `m` - Length of bit array
    /// * `k` - Number of hash functions to use
    /// * `p` - False Positive probability
    ///
    pub fn calc_number_of_items(m: u64, k: u32, p: f64) -> u64 {
        let k_float = k as f64;
        (m as f64 / (-(k_float) / (1.0_f64 - (p.log(E) / k_float).exp()).log(E))).ceil() as u64
    }

    /// Calculates item size and hash count
    /// by increasing the hash_count to fit the maximum possible number of items.
    ///
    /// # Arguments
    /// * `hash_count` - Number of hash functions to use
    /// * `fp_prob` - False Positive probability in decimal
    ///
    pub fn calc_item_size_and_hash_count(size: u64, fp_prob: f64) -> (u64, u32) {
        let mut number_of_items: u64 = 0;
        for k in 1..=u32::MAX {
            let temp_item_size = Self::calc_number_of_items(size, k, fp_prob); // ceil(m / (-k / log(1 - exp(log(p) / k))))
            if number_of_items > temp_item_size {
                return (number_of_items, k - 1);
            } else {
                number_of_items = temp_item_size;
            }
        }
        (number_of_items, u32::MAX)
    }

    /// Rounds the given length to the nearest multiple of T (size of the BitStore type in bits)
    ///
    /// # Arguments
    /// * `length` - Length to round
    ///   
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
    pub fn add_aliased<I>(&self, item: &mut I) -> Result<(), BloomFilterError>
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

impl<T> Display for BloomFilter<T>
where
    T: BitStore,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BloomFilter(designed for {} items, size {} bytes, false positive probability {}, using {} hash functions)",
            self.number_of_items,
            self.size(),
            self.fp_prob,
            self.hash_count
        )
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

    #[test]
    fn test_rounding() {
        assert_eq!(BloomFilter::<u8>::round_to_t(1), 8);
        assert_eq!(BloomFilter::<u8>::round_to_t(7), 8);
        assert_eq!(BloomFilter::<u8>::round_to_t(8), 8);
        assert_eq!(BloomFilter::<u8>::round_to_t(9), 16);

        assert_eq!(BloomFilter::<u16>::round_to_t(1), 16);
        assert_eq!(BloomFilter::<u16>::round_to_t(15), 16);
        assert_eq!(BloomFilter::<u16>::round_to_t(16), 16);
        assert_eq!(BloomFilter::<u16>::round_to_t(17), 32);

        assert_eq!(BloomFilter::<u32>::round_to_t(1), 32);
        assert_eq!(BloomFilter::<u32>::round_to_t(31), 32);
        assert_eq!(BloomFilter::<u32>::round_to_t(32), 32);
        assert_eq!(BloomFilter::<u32>::round_to_t(33), 64);

        assert_eq!(BloomFilter::<u64>::round_to_t(1), 64);
        assert_eq!(BloomFilter::<u64>::round_to_t(63), 64);
        assert_eq!(BloomFilter::<u64>::round_to_t(64), 64);
        assert_eq!(BloomFilter::<u64>::round_to_t(65), 128);

        assert_eq!(
            BloomFilter::<usize>::round_to_t(1),
            std::mem::size_of::<usize>() as u64 * 8
        );
        assert_eq!(
            BloomFilter::<usize>::round_to_t(63),
            std::mem::size_of::<usize>() as u64 * 8
        );
        assert_eq!(
            BloomFilter::<usize>::round_to_t(64),
            std::mem::size_of::<usize>() as u64 * 8
        );
        assert_eq!(
            BloomFilter::<usize>::round_to_t(65),
            std::mem::size_of::<usize>() as u64 * 8 * 2
        );
    }

    /// Test calculation of length
    #[test]
    fn test_calc_length() {
        let length = BloomFilter::<u8>::calc_length(80_000_000, 0.001);
        assert_eq!(length, 1150207006);
        assert_ne!(length % 8, 0);

        let length_rounded = BloomFilter::<u8>::calc_length_rounded(80_000_000, 0.001);
        assert_eq!(length_rounded % 8, 0)
    }

    #[test]
    fn test_calc_hash_count() {
        let length = BloomFilter::<u8>::calc_length(80_000_000, 0.001);
        let hash_count = BloomFilter::<u8>::calc_hash_count(length, 80_000_000).unwrap();
        assert_eq!(hash_count, 10);
    }

    /// Test display implementation
    #[test]
    fn test_dispaly() {
        let bloom_filter: BloomFilter<u8> =
            BloomFilter::new_by_item_count_and_fp_prob(1000, 0.01).unwrap();
        let display = format!("{}", bloom_filter);
        assert_eq!(display, "BloomFilter(designed for 1000 items, size 1199 bytes, false positive probability 0.01, using 6 hash functions)");
    }
}
