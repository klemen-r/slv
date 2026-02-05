//! Error types for SLV Options

use thiserror::Error;

#[derive(Error, Debug)]
pub enum SLVError {
    #[error("Data error: {0}")]
    Data(String),

    #[error("Calibration error: {0}")]
    Calibration(String),

    #[error("Pricing error: {0}")]
    Pricing(String),

    #[error("Numerical error: {0}")]
    Numerical(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

pub type SLVResult<T> = Result<T, SLVError>;

impl SLVError {
    pub fn data(msg: impl Into<String>) -> Self {
        Self::Data(msg.into())
    }

    pub fn calibration(msg: impl Into<String>) -> Self {
        Self::Calibration(msg.into())
    }

    pub fn pricing(msg: impl Into<String>) -> Self {
        Self::Pricing(msg.into())
    }

    pub fn numerical(msg: impl Into<String>) -> Self {
        Self::Numerical(msg.into())
    }

    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }
}
