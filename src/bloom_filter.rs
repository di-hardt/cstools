/// Error handling
pub mod error;

/// Implementation for HDF5 serialization and deserialization
#[cfg(feature = "hdf5")]
pub mod hdf5_utils;

/// Implementation for Serde serialization and deserialization
#[cfg(feature = "serde")]
pub mod serde_utils;

use std::{fmt::Display, io::Read};

use bitvec::prelude::*;
use murmur3::murmur3_x64_128 as murmur3hash;

use crate::bloom_filter::error::BloomFilterError;

pub trait IsBuilder<T>
where
    T: BitStore,
{
    /// Calculates the length of the bit array `m`
    ///
    /// $$ m = -\frac{n ln(p)}{ln(2)^2}  $$
    ///
    /// # Arguments
    ///
    /// `n` - Number of items expected to be stored in filter
    /// `p` - False Positive probability in decimal
    ///
    fn calc_length(n: u64, p: f64) -> u64 {
        let n = n as f64;
        -((n * p.ln()) / 2_f64.ln().powf(2.0)).ceil() as u64
    }

    /// Calculates the false positive probability `f` using
    /// the following formula
    ///
    /// $$ (1 − e^(−kn/m))^k $$
    ///
    /// # Arguments
    ///
    /// * `m` - Length of bit array
    /// * `n` - Number of items expected to be stored in filter
    /// * `k` - Number of hash functions to use
    ///
    fn calc_false_positive_prob(m: u64, n: u64, k: u32) -> f64 {
        let k = k as f64;
        let n = n as f64;
        let m = m as f64;

        (1.0 - ((-k * n) / m).exp()).powf(k)
    }

    /// Calculates the number of hash function `k` to apply when checking for an item, using
    /// following formula
    ///  $$ k = \frac{m}{n} * ln(2) $$
    ///
    /// # Arguments
    ///
    /// * `m` - Length of bit array
    /// * `n` - Number of items expected to be stored in filter
    ///
    fn calc_hash_count(m: u64, n: u64) -> Result<u32, BloomFilterError> {
        let m = m as f64;
        let n = n as f64;

        let k = ((m / n) * 2.0_f64.ln()).round();
        if k > u32::MAX as f64 {
            return Err(BloomFilterError::HashCountTooLarge);
        }
        Ok(k as u32)
    }

    /// Calculates the maximum number of items `n` the filter can hold
    ///
    /// $$ n = -\frac{m * ln(2)^2}{ln(p)} $$
    ///
    /// # Arguments
    /// * `m` - Length of bit array
    /// * `p` - False Positive probability
    ///
    fn calc_number_of_items(m: u64, p: f64) -> u64 {
        let m = m as f64;
        -((m * 2_f64.ln().powf(2.0)) / p.ln()).ceil() as u64
    }

    /// Rounds the given length to the nearest multiple of T (size of the BitStore type in bits)
    ///
    /// # Arguments
    /// * `length` - Length to round
    ///   
    fn round_to_t(length: u64) -> u64 {
        let remains = length % (std::mem::size_of::<T>() * 8) as u64;
        if remains == 0 {
            length
        } else {
            length + (std::mem::size_of::<T>() * 8) as u64 - remains
        }
    }
}

pub struct Builder<T>
where
    T: BitStore,
{
    phantom: std::marker::PhantomData<T>,
}

impl<T> Builder<T>
where
    T: BitStore,
{
    pub fn with_false_positive_probability(
        self,
        false_positive_probability: f64,
    ) -> FalsePositiveProbabilityBuilder<T> {
        FalsePositiveProbabilityBuilder {
            false_positive_probability,
            phantom: std::marker::PhantomData,
        }
    }

    /// Sets the number of items expected to be stored in the bloom filter
    ///
    /// # Arguments
    /// * `number_of_items` - Number of items expected to be stored in filter
    pub fn with_length(self, length: u64) -> LengthBuilder<T> {
        LengthBuilder {
            length,
            phantom: std::marker::PhantomData,
        }
    }

    /// Sets the number of items expected to be stored in the bloom filter
    ///
    /// # Arguments
    /// * `number_of_items` - Number of items expected to be stored in filter
    pub fn with_number_of_items(self, number_of_items: u64) -> NumberOfItemsBuilder<T> {
        NumberOfItemsBuilder {
            number_of_items,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T> IsBuilder<T> for Builder<T> where T: BitStore {}

pub struct FalsePositiveProbabilityBuilder<T>
where
    T: BitStore,
{
    false_positive_probability: f64,
    phantom: std::marker::PhantomData<T>,
}

impl<T> FalsePositiveProbabilityBuilder<T>
where
    T: BitStore,
{
    /// Sets the length of the bloom filter in bits
    ///
    /// # Arguments
    /// * `length` - Length of the bloom filter in bits
    ///
    pub fn with_length(self, length: u64) -> Result<BloomFilter<T>, BloomFilterError> {
        if self.false_positive_probability == 0.0 {
            return Err(BloomFilterError::FalsePositiveProbabilityOne);
        }

        if self.false_positive_probability == 1.0 {
            return Err(BloomFilterError::FalsePositiveProbabilityOne);
        }

        if length == 0 {
            return Err(BloomFilterError::LengthZero);
        }

        let rounded_length = Self::round_to_t(length);

        let number_of_items =
            Self::calc_number_of_items(rounded_length, self.false_positive_probability);

        if number_of_items == 0 {
            return Err(BloomFilterError::NumberOfItemsZero);
        }

        let hash_count = Self::calc_hash_count(rounded_length, number_of_items)?;

        Ok(BloomFilter::new(
            self.false_positive_probability,
            hash_count,
            number_of_items,
            rounded_length,
        ))
    }

    /// Sets the number of items expected to be stored in the bloom filter
    ///
    /// # Arguments
    /// * `number_of_items` - Number of items expected to be stored in filter
    ///
    pub fn with_number_of_items(
        self,
        number_of_items: u64,
    ) -> Result<BloomFilter<T>, BloomFilterError> {
        if self.false_positive_probability == 1.0 {
            return Err(BloomFilterError::FalsePositiveProbabilityOne);
        }

        if number_of_items == 0 {
            return Err(BloomFilterError::NumberOfItemsZero);
        }

        // length of bit array to use
        let length = Self::round_to_t(Self::calc_length(
            number_of_items,
            self.false_positive_probability,
        ));

        if length == 0 {
            return Err(BloomFilterError::LengthZero);
        }

        // Number of hash functions to use
        let hash_count = Self::calc_hash_count(length, number_of_items)?;

        Ok(BloomFilter::new(
            self.false_positive_probability,
            hash_count,
            number_of_items,
            length,
        ))
    }
}

impl<T> IsBuilder<T> for FalsePositiveProbabilityBuilder<T> where T: BitStore {}

pub struct LengthBuilder<T>
where
    T: BitStore,
{
    length: u64,
    phantom: std::marker::PhantomData<T>,
}

impl<T> LengthBuilder<T>
where
    T: BitStore,
{
    /// Sets the false positive probability
    ///
    /// # Arguments
    /// * `false_positive_probability` - False_positive_probability to use
    ///
    pub fn with_false_positive_probability(
        self,
        false_positive_probability: f64,
    ) -> Result<BloomFilter<T>, BloomFilterError> {
        FalsePositiveProbabilityBuilder {
            false_positive_probability,
            phantom: std::marker::PhantomData,
        }
        .with_length(self.length)
    }

    /// Sets the number of expected items
    ///
    /// # Arguments
    /// * `number_of_items` - Number of expected items
    ///
    pub fn with_number_of_items(
        self,
        number_of_items: u64,
    ) -> Result<BloomFilter<T>, BloomFilterError> {
        if self.length == 0 {
            return Err(BloomFilterError::LengthZero);
        }

        if number_of_items == 0 {
            return Err(BloomFilterError::NumberOfItemsZero);
        }

        let hash_count = Self::calc_hash_count(self.length, number_of_items)?;

        let false_positive_probability =
            Self::calc_false_positive_prob(self.length, number_of_items, hash_count);

        Ok(BloomFilter::new(
            false_positive_probability,
            hash_count,
            number_of_items,
            self.length,
        ))
    }
}

impl<T> IsBuilder<T> for LengthBuilder<T> where T: BitStore {}

pub struct NumberOfItemsBuilder<T>
where
    T: BitStore,
{
    number_of_items: u64,
    phantom: std::marker::PhantomData<T>,
}

impl<T> NumberOfItemsBuilder<T>
where
    T: BitStore,
{
    /// Sets the false positive probability in decimal
    ///
    /// # Arguments
    /// * `false_positive_probability` - False Positive probability in decimal
    ///
    pub fn with_false_positive_probability(
        self,
        false_positive_probability: f64,
    ) -> Result<BloomFilter<T>, BloomFilterError> {
        FalsePositiveProbabilityBuilder::<T> {
            false_positive_probability,
            phantom: std::marker::PhantomData,
        }
        .with_number_of_items(self.number_of_items)
    }

    /// Sets the length of the bloom filter in bits
    ///
    /// # Argument
    /// * `length` - Length of the Bloom Filter in bits
    ///
    pub fn with_length(self, length: u64) -> Result<BloomFilter<T>, BloomFilterError> {
        LengthBuilder {
            length,
            phantom: std::marker::PhantomData,
        }
        .with_number_of_items(self.number_of_items)
    }
}

impl<T> IsBuilder<T> for NumberOfItemsBuilder<T> where T: BitStore {}

/// BloomFilter using murmur3 hash functions
///
#[derive(Clone)]
pub struct BloomFilter<T>
where
    T: BitStore,
{
    /// False positive probability
    pub(crate) false_positive_probability: f64,

    /// Length of the bloom filter as u128 fosr position calculation
    pub(crate) length: u128,

    /// Number of items the filter is designed to hold
    pub(crate) number_of_items: u64,

    /// Number of hash functions to apply
    pub(crate) hash_count: u32,

    // Bit vector
    pub(crate) bitvec: BitVec<T, Msb0>,
}

impl<T> BloomFilter<T>
where
    T: BitStore,
{
    pub fn build() -> Builder<T> {
        Builder {
            phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new Bloom filter
    ///
    /// Arguments:
    /// * `false_positive_probability` - False Positive probability in decimal
    /// * `hash_count` - Number of hash functions to use
    /// * `number_of_items` - Number of items expected to be stored in filter
    /// * `length` - Length of the bloom filter in bits
    ///
    pub(crate) fn new(
        false_positive_probability: f64,
        hash_count: u32,
        number_of_items: u64,
        length: u64,
    ) -> Self {
        let bitvec = bitvec!(T, Msb0; 0; length as usize);
        let length = length as u128;

        Self {
            false_positive_probability,
            hash_count,
            number_of_items,
            length,
            bitvec,
        }
    }

    /// Creates a new Bloom filter
    ///
    /// Arguments:
    /// * `false_positive_probability` - False Positive probability in decimal
    /// * `hash_count` - Number of hash functions to use
    /// * `number_of_items` - Number of items expected to be stored in filter
    /// * `bitvec` - Bit vector to use
    ///
    #[cfg(any(feature = "serde", feature = "hdf5"))]
    pub(crate) fn new_with_bitvec(
        false_positive_probability: f64,
        hash_count: u32,
        number_of_items: u64,
        bitvec: BitVec<T, Msb0>,
    ) -> Self {
        let length = bitvec.len() as u128;

        Self {
            false_positive_probability,
            hash_count,
            number_of_items,
            length,
            bitvec,
        }
    }

    /// Get false positive probability
    ///
    pub fn false_positive_probability(&self) -> f64 {
        self.false_positive_probability
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
    pub fn bitvec(&self) -> &BitVec<T, Msb0> {
        &self.bitvec
    }

    /// Length of bit vector
    ///
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.bitvec.len()
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
            self.false_positive_probability,
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

        let mut bloom_filter = BloomFilter::<T>::build()
            .with_number_of_items(some_strings.len() as u64)
            .with_false_positive_probability(0.01)
            .unwrap();

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

        let bloom_filter = BloomFilter::<T>::build()
            .with_number_of_items(some_strings.len() as u64)
            .with_false_positive_probability(0.01)
            .unwrap();

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
        assert_eq!(Builder::<u8>::round_to_t(1), 8);
        assert_eq!(Builder::<u8>::round_to_t(7), 8);
        assert_eq!(Builder::<u8>::round_to_t(8), 8);
        assert_eq!(Builder::<u8>::round_to_t(9), 16);

        assert_eq!(Builder::<u16>::round_to_t(1), 16);
        assert_eq!(Builder::<u16>::round_to_t(15), 16);
        assert_eq!(Builder::<u16>::round_to_t(16), 16);
        assert_eq!(Builder::<u16>::round_to_t(17), 32);

        assert_eq!(Builder::<u32>::round_to_t(1), 32);
        assert_eq!(Builder::<u32>::round_to_t(31), 32);
        assert_eq!(Builder::<u32>::round_to_t(32), 32);
        assert_eq!(Builder::<u32>::round_to_t(33), 64);

        assert_eq!(Builder::<u64>::round_to_t(1), 64);
        assert_eq!(Builder::<u64>::round_to_t(63), 64);
        assert_eq!(Builder::<u64>::round_to_t(64), 64);
        assert_eq!(Builder::<u64>::round_to_t(65), 128);

        assert_eq!(
            Builder::<usize>::round_to_t(1),
            std::mem::size_of::<usize>() as u64 * 8
        );
        assert_eq!(
            Builder::<usize>::round_to_t(63),
            std::mem::size_of::<usize>() as u64 * 8
        );
        assert_eq!(
            Builder::<usize>::round_to_t(64),
            std::mem::size_of::<usize>() as u64 * 8
        );
        assert_eq!(
            Builder::<usize>::round_to_t(65),
            std::mem::size_of::<usize>() as u64 * 8 * 2
        );
    }

    // This takes a long time to run compared to other tests so we just test it for u8
    #[test]
    fn test_false_positive_probability_zero() {
        let some_strings: Vec<String> =
            read_to_string(PathBuf::from("test_data/10000_random_strings.txt"))
                .unwrap()
                .lines()
                .map(String::from)
                .collect();

        let mut bloom_filter = BloomFilter::<u8>::build()
            .with_length(1024_u64.pow(3)) // 1 GiB
            .with_number_of_items(some_strings.len() as u64)
            .unwrap();

        assert_eq!(bloom_filter.false_positive_probability(), 0.0);

        for a_string in some_strings[1000..].iter() {
            bloom_filter
                .add(&mut Cursor::new(a_string.as_bytes()))
                .unwrap();
        }

        for a_string in some_strings[1000..].iter() {
            assert!(bloom_filter
                .contains(&mut Cursor::new(a_string.as_bytes()))
                .unwrap());
        }

        for a_string in some_strings[..1000].iter() {
            assert!(!bloom_filter
                .contains(&mut Cursor::new(a_string.as_bytes()))
                .unwrap(),);
        }
    }

    /// Test calculation of length
    #[test]
    fn test_calc_length() {
        let length = Builder::<u8>::calc_length(80_000_000, 0.001);
        assert_eq!(length, 1150207005);
        assert_ne!(length % 8, 0);

        let length_rounded = Builder::<u8>::round_to_t(length);
        assert_eq!(length_rounded % 8, 0)
    }

    #[test]
    fn test_calc_hash_count() {
        let length = Builder::<u8>::calc_length(80_000_000, 0.001);
        let hash_count = Builder::<u8>::calc_hash_count(length, 80_000_000).unwrap();
        assert_eq!(hash_count, 10);
    }
}
