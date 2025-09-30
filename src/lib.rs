// # Create absolute path to readme ti increase compatible for different build targets
//  https://gist.github.com/JakeHartnell/2c1fa387f185f5dc46c9429470a2e2be
#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/Readme.md"))]

/// BloomFilter implementation
pub mod bloom_filter;
/// Helpers for HDF5 serialization
#[cfg(feature = "hdf5")]
pub mod hdf5_utils;
#[cfg(feature = "serde")]
pub mod serde_utils;
