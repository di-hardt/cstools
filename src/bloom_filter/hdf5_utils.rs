use std::cell::Cell;
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, AtomicUsize};

use bitvec::boxed::BitBox;
use bitvec::order::Msb0;
use bitvec::store::BitStore;

use crate::bloom_filter::error::{BloomFilterHdf5Error, BloomFilterSerDesError};
use crate::bloom_filter::BloomFilter;

/// Trait for converting bit storage types to bytes
pub trait BitStoreToBytes: BitStore {
    fn as_bytes(&self) -> Vec<u8>;
}

macro_rules! uint_to_bytes {
    ($($t:ty),+ $(,)?) => { $(
        impl BitStoreToBytes for $t {
            fn as_bytes(&self) -> Vec<u8> {
                self.to_le_bytes().to_vec()
            }
        }
    )+ };
}

macro_rules! cell_uint_to_bytes {
    ($($t:ty),+ $(,)?) => { $(
        impl BitStoreToBytes for $t {
            fn as_bytes(&self) -> Vec<u8> {
                self.get().to_le_bytes().to_vec()
            }
        }
    )+ };
}

macro_rules! atomic_uint_to_bytes {
    ($($t:ty),+ $(,)?) => { $(
        impl BitStoreToBytes for $t {
            fn as_bytes(&self) -> Vec<u8> {
                self.load(std::sync::atomic::Ordering::Relaxed)
                    .to_le_bytes()
                    .to_vec()
            }
        }
    )+ };
}

uint_to_bytes!(u8, u16, u32, u64, usize);
cell_uint_to_bytes!(Cell<u8>, Cell<u16>, Cell<u32>, Cell<u64>, Cell<usize>);
atomic_uint_to_bytes!(AtomicU8, AtomicU16, AtomicU32, AtomicU64, AtomicUsize);

/// Trait for creating vectors of bit storage types from bytes
pub trait BitStoreVecFromBytes: Sized + BitStore {
    const SIZE: usize;
    fn from_bytes(bytes: Vec<u8>) -> Vec<Self>;
}

macro_rules! uint_vec_from_bytes {
    ($(($target:ty, $primitive:ty)),+ $(,)?) => { $(
        impl BitStoreVecFromBytes for $target {
            const SIZE: usize = std::mem::size_of::<$primitive>();
            fn from_bytes(bytes: Vec<u8>) -> Vec<Self> {
                bytes
                    .chunks_exact(Self::SIZE)
                    .map(|chunk| {
                        let mut buffer = [0u8; Self::SIZE];
                        buffer.copy_from_slice(chunk);
                        Self::from(<$primitive>::from_le_bytes(buffer))
                    })
                    .collect()
            }
        }
    )+ };
}

uint_vec_from_bytes!((u8, u8), (u16, u16), (u32, u32), (u64, u64), (usize, usize));
uint_vec_from_bytes!(
    (Cell<u8>, u8),
    (Cell<u16>, u16),
    (Cell<u32>, u32),
    (Cell<u64>, u64),
    (Cell<usize>, usize)
);

uint_vec_from_bytes!(
    (AtomicU8, u8),
    (AtomicU16, u16),
    (AtomicU32, u32),
    (AtomicU64, u64),
    (AtomicUsize, usize)
);

impl<T> BloomFilter<T>
where
    T: BitStore + BitStoreToBytes + BitStoreVecFromBytes,
{
    /// Loads bloom filter from hdf5 file
    ///
    /// # Arguments
    /// * `path` - Path to hdf5 file
    ///
    pub fn load_hdf5(path: &std::path::PathBuf) -> Result<Self, BloomFilterHdf5Error> {
        use bitvec::{order::Msb0, vec::BitVec};

        let file = hdf5::File::open(path)?;
        let type_memory_width = file.dataset("type_memory_width")?.read_scalar::<u8>()?;
        if type_memory_width as usize != std::mem::size_of::<T>() {
            return Err(BloomFilterSerDesError::MemoryLayoutMismatch(
                type_memory_width,
                std::mem::size_of::<T>() as u8,
            )
            .into());
        }
        let hash_count = file.dataset("hash_count")?.read_scalar::<u32>()?;
        let fp_prob = file.dataset("fp_prob")?.read_scalar::<f64>()?;
        let bytes = Self::decode_hex(
            file.dataset("bit_array")?
                .read_scalar::<hdf5::types::VarLenAscii>()?
                .as_str(),
        )?;
        Ok(Self::new(
            fp_prob,
            hash_count,
            BitVec::<T, Msb0>::from_slice(&bytes).into_boxed_bitslice(),
        ))
    }

    /// Saves bloom filter to hdf5 file
    ///
    /// # Arguments
    /// * `path` - Path to hdf5 file
    ///
    pub fn save_hdf5(&self, path: &std::path::PathBuf) -> Result<(), BloomFilterHdf5Error> {
        let file = hdf5::File::create(path)?;
        file.new_dataset::<u8>()
            .create("type_memory_width")?
            .write_scalar(&(std::mem::size_of::<T>() as u8))?;
        file.new_dataset::<u32>()
            .create("hash_count")?
            .write_scalar(&self.hash_count)?;
        file.new_dataset::<f64>()
            .create("fp_prob")?
            .write_scalar(&self.fp_prob)?;
        // Convert bitvec to hex string
        let s_ascii = Self::encode_hex(&self.bitvec);
        // Save hex string to hdf5 file
        file.new_dataset::<hdf5::types::VarLenAscii>()
            .create("bit_array")?
            .write_scalar(
                &hdf5::types::VarLenAscii::from_ascii(&s_ascii)
                    .map_err(BloomFilterHdf5Error::HexToVarAscii)?,
            )?;
        Ok(())
    }

    /// Decodes hex string to bytes
    ///
    /// # Arguments
    /// * `s` - Hex string
    ///
    pub fn decode_hex(s: &str) -> Result<Vec<T>, BloomFilterHdf5Error> {
        let bytes = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
            .collect::<Result<Vec<u8>, core::num::ParseIntError>>()
            .map_err(BloomFilterHdf5Error::HexDecode)?;
        Ok(T::from_bytes(bytes))
    }

    /// Encodes bit array to hex string
    ///
    /// # Arguments
    /// * `bit_array` - Bit array
    ///
    pub fn encode_hex(bit_array: &BitBox<T, Msb0>) -> String {
        bit_array
            .as_raw_slice()
            .iter()
            .flat_map(|b| b.as_bytes())
            .map(|b| format!("{:02X}", b))
            .collect::<String>()
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::read_to_string, io::Cursor, path::PathBuf};

    use super::*;

    #[cfg(feature = "hdf5")]
    #[test]
    fn test_hdf5() {
        test_hdf5_generic::<u8>();
        test_hdf5_generic::<u16>();
        test_hdf5_generic::<u32>();
        test_hdf5_generic::<u64>();
        test_hdf5_generic::<usize>();

        test_hdf5_generic::<Cell<u8>>();
        test_hdf5_generic::<Cell<u16>>();
        test_hdf5_generic::<Cell<u32>>();
        test_hdf5_generic::<Cell<u64>>();
        test_hdf5_generic::<Cell<usize>>();

        test_hdf5_generic::<AtomicU8>();
        test_hdf5_generic::<AtomicU16>();
        test_hdf5_generic::<AtomicU32>();
        test_hdf5_generic::<AtomicU64>();
    }

    fn test_hdf5_generic<T>()
    where
        T: BitStore + BitStoreToBytes + BitStoreVecFromBytes,
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

        let temp_file = std::env::temp_dir().join("bloom_filter.h5");
        if temp_file.is_file() {
            std::fs::remove_file(&temp_file).unwrap();
        }

        bloom_filter.save_hdf5(&temp_file).unwrap();

        let read_bloom_filter: BloomFilter<T> = BloomFilter::load_hdf5(&temp_file).unwrap();

        assert_eq!(bloom_filter.len(), read_bloom_filter.len());
        assert_eq!(bloom_filter.hash_count, read_bloom_filter.hash_count);
        assert_eq!(bloom_filter.fp_prob, read_bloom_filter.fp_prob);
        assert_eq!(bloom_filter.bitvec, read_bloom_filter.bitvec);

        for a_string in some_strings.iter() {
            assert!(read_bloom_filter
                .contains(&mut Cursor::new(a_string.as_bytes()))
                .unwrap());
        }

        if temp_file.is_file() {
            std::fs::remove_file(&temp_file).unwrap();
        }
    }
}
