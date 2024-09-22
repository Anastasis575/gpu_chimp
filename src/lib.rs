use wgpu::{Device,Queue};
pub  struct Context{
    device:Device,
    queue:Queue
}

impl Context {
    pub fn new(device: Device, queue: Queue) -> Self {
        Self { device, queue }
    }

    pub fn device(&self)->&Device{
        &self.device
    }
    pub fn queue(&self)->&Queue{
        &self.queue
    }
    pub fn device_mut(&mut self)->&mut Device{
        &mut self.device
    }
    pub fn queue_mut(&mut self)->&mut Queue{
        &mut self.queue
    }
}


pub fn chimpCompress(values:&[f32])->Result<[u8]>{

}
// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
