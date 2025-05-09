# Compress Library

## Description

A GPU implementation of the

## Getting Started

To build the project you need rust with cargo version 1.80.1

## Running

In order to run the main tests you can use the following

```shell
    cargo test --bin compressapi 
    
    #Or a specific test with it's function name
    cargo test --bin compressapi compress_test::wgpu_compress_32
    
    # to enable the logging messages (by project)
    RUST_LOG=compressapi,wgpu_compress_32 cargo test --bin compressapi
    
    # by default the default break up of values to wgsl workgroups is 64,
    # to change it pass the CHIMP_BUFFER_SIZE environment variable (has to be <=256)
    CHIMP_BUFFER_SIZE=256 cargo test --bin compressapi
```

The above is applicable on linux/mac on windows the same would be possible with

```shell 
    $env:CHIMP_BUFFER_SIZE=256; cargo test --bin compressapi
```

uniformly you can use an .env file like so

```yaml
    CHIMP_BUFFER_SIZE=256
```