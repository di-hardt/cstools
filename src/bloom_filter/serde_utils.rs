use std::fmt;

use bitvec::store::BitStore;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};

use crate::bloom_filter::error::BloomFilterSerDesError;
use crate::bloom_filter::BloomFilter;

impl<T> Serialize for BloomFilter<T>
where
    T: BitStore + Serialize,
    T::Mem: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("BloomFilter", 5)?;
        state.serialize_field("type_memory_width", &std::mem::size_of::<T>())?;
        state.serialize_field("hash_count", &self.hash_count())?;
        state.serialize_field(
            "false_positive_probability",
            &self.false_positive_probability,
        )?;
        state.serialize_field("number_of_items", &self.number_of_items)?;
        state.serialize_field("bit_array", &self.bitvec)?;
        state.end()
    }
}

impl<'de, T> Deserialize<'de> for BloomFilter<T>
where
    T: BitStore + Deserialize<'de>,
    T::Mem: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            TypeMemoryWidth,
            HashCount,
            FalsePositiveProbability,
            NumberOfItems,
            BitArray,
        }
        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str(
                            "`type_memory_width`, `hash_count`, `false_positive_probability`, `number_of_items`, or `bit_array`",
                        )
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "type_memory_width" => Ok(Field::TypeMemoryWidth),
                            "hash_count" => Ok(Field::HashCount),
                            "false_positive_probability" => Ok(Field::FalsePositiveProbability),
                            "number_of_items" => Ok(Field::NumberOfItems),
                            "bit_array" => Ok(Field::BitArray),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct BloomFilterVisitor<'de, T>
        where
            T: BitStore + Deserialize<'de>,
            T::Mem: Deserialize<'de>,
        {
            _lifteime_marker: std::marker::PhantomData<&'de T>,
        }

        impl<'de, T> Visitor<'de> for BloomFilterVisitor<'de, T>
        where
            T: BitStore + Deserialize<'de>,
            T::Mem: Deserialize<'de>,
        {
            type Value = BloomFilter<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct BloomFilter")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<BloomFilter<T>, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let type_memory_width: u8 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                if type_memory_width != std::mem::size_of::<T>() as u8 {
                    return Err(de::Error::custom(
                        BloomFilterSerDesError::MemoryLayoutMismatch(
                            type_memory_width,
                            std::mem::size_of::<T>() as u8,
                        ),
                    ));
                }

                let hash_count = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let false_positive_probability = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;
                let number_of_items = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(3, &self))?;
                let bit_array = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(4, &self))?;
                Ok(BloomFilter::new(
                    false_positive_probability,
                    hash_count,
                    number_of_items,
                    bit_array,
                ))
            }

            fn visit_map<V>(self, mut map: V) -> Result<BloomFilter<T>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut type_memory_width: Option<u8> = None;
                let mut hash_count = None;
                let mut false_positive_probability = None;
                let mut number_of_items = None;
                let mut bit_array = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::TypeMemoryWidth => {
                            if type_memory_width.is_some() {
                                return Err(de::Error::duplicate_field("type_memory_width"));
                            }
                            type_memory_width = Some(map.next_value()?);
                            if type_memory_width.unwrap() != std::mem::size_of::<T>() as u8 {
                                return Err(de::Error::custom(
                                    BloomFilterSerDesError::MemoryLayoutMismatch(
                                        type_memory_width.unwrap(),
                                        std::mem::size_of::<T>() as u8,
                                    ),
                                ));
                            }
                        }
                        Field::HashCount => {
                            if hash_count.is_some() {
                                return Err(de::Error::duplicate_field("hash_count"));
                            }
                            hash_count = Some(map.next_value()?);
                        }

                        Field::FalsePositiveProbability => {
                            if false_positive_probability.is_some() {
                                return Err(de::Error::duplicate_field(
                                    "false_positive_probability",
                                ));
                            }
                            false_positive_probability = Some(map.next_value()?);
                        }

                        Field::NumberOfItems => {
                            if number_of_items.is_some() {
                                return Err(de::Error::duplicate_field("number_of_items"));
                            }
                            number_of_items = Some(map.next_value()?);
                        }

                        Field::BitArray => {
                            if bit_array.is_some() {
                                return Err(de::Error::duplicate_field("bit_array"));
                            }
                            bit_array = Some(map.next_value()?);
                        }
                    }
                }
                if type_memory_width.is_none() {
                    return Err(de::Error::missing_field("type_memory_width"));
                }

                Ok(BloomFilter::new(
                    false_positive_probability
                        .ok_or_else(|| de::Error::missing_field("false_positive_probability"))?,
                    hash_count.ok_or_else(|| de::Error::missing_field("hash_count"))?,
                    number_of_items.ok_or_else(|| de::Error::missing_field("number_of_items"))?,
                    bit_array.ok_or_else(|| de::Error::missing_field("bit_array"))?,
                ))
            }
        }

        const FIELDS: &[&str] = &[
            "type_memory_width",
            "hash_count",
            "false_positive_probability",
            "number_of_items",
            "bit_array",
        ];
        deserializer.deserialize_struct(
            "Duration",
            FIELDS,
            BloomFilterVisitor {
                _lifteime_marker: std::marker::PhantomData,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, AtomicUsize};
    use std::{fs::read_to_string, path::PathBuf};

    use super::*;

    /// Using Serde to serialize and deserialize the bloom filter
    ///
    #[cfg(feature = "serde")]
    #[test]
    fn test_serde() {
        test_serde_generic::<u8>();
        test_serde_generic::<u16>();
        test_serde_generic::<u32>();
        test_serde_generic::<u64>();
        test_serde_generic::<usize>();

        test_serde_generic::<Cell<u8>>();
        test_serde_generic::<Cell<u16>>();
        test_serde_generic::<Cell<u32>>();
        test_serde_generic::<Cell<u64>>();
        test_serde_generic::<Cell<usize>>();

        test_serde_generic::<AtomicU8>();
        test_serde_generic::<AtomicU16>();
        test_serde_generic::<AtomicU32>();
        test_serde_generic::<AtomicU64>();
        test_serde_generic::<AtomicUsize>();
    }

    fn test_serde_generic<T>()
    where
        T: BitStore + Serialize + for<'de> Deserialize<'de>,
        T::Mem: Serialize + for<'de> Deserialize<'de>,
    {
        use rmp_serde::{Deserializer, Serializer};
        use serde::{Deserialize, Serialize};

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
            bloom_filter.add(a_string.as_bytes()).unwrap();
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
        let read_bloom_filter: BloomFilter<T> =
            BloomFilter::deserialize(&mut Deserializer::new(&mut byte_reader)).unwrap();

        assert!(bloom_filter.len() == read_bloom_filter.len());
        assert!(bloom_filter.hash_count == read_bloom_filter.hash_count);
        assert!(
            bloom_filter.false_positive_probability == read_bloom_filter.false_positive_probability
        );
        assert!(bloom_filter.bitvec == read_bloom_filter.bitvec);

        for a_string in some_strings.iter() {
            assert!(read_bloom_filter.contains(a_string.as_bytes()).unwrap());
        }

        if temp_file_path.is_file() {
            std::fs::remove_file(&temp_file_path).unwrap();
        }
    }
}
