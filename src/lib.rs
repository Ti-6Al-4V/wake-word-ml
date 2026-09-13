// Общая библиотека проекта: всё, что нужно нескольким бинарям.
//   audio   — WAV, дизер, подгонка окна
//   mfcc    — признаки
//   dataset — сплиты и загрузка
//   model   — архитектура CNN (train / eval / export смотрят в одно место)

pub mod audio;
pub mod dataset;
pub mod mfcc;
pub mod model;
