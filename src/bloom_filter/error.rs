#[cfg(feature = "hdf5")]
use hdf5::types::StringError;
use thiserror::Error;

/// Error for bloom filter operations
#[derive(Error, Debug)]
pub enum BloomFilterError {
    #[error("Length is zero")]
    LengthZero,
    #[error("Hash count too large (2^32)")]
    HashCountTooLarge,
    #[error("Hash error: {0}")]
    Hashing(std::io::Error),
}

/// Common error for serialization and deserialization
#[derive(Error, Debug)]
pub enum BloomFilterSerDesError {
    #[error("Mismatch bewteen stored memory layout ({0} bytes) and chosen type ({1} bytes)")]
    MemoryLayoutMismatch(u8, u8),
}

/// Errors for HDF5 serialization and deserialization
#[cfg(feature = "hdf5")]
#[derive(Error, Debug)]
pub enum BloomFilterHdf5Error {
    #[error("{0}")]
    BloomFilter(#[from] BloomFilterError),
    #[error("{0}")]
    BloomFilterSerDesError(#[from] BloomFilterSerDesError),
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),
    #[error("Unable to decode hex string: `{0}`")]
    HexDecode(std::num::ParseIntError),
    #[error("Unable to convert hex striong into var ascii string: `{0}`")]
    HexToVarAscii(StringError),
}
