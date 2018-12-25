# Prediksi Harga Minyak Kelapa Sawit Menggunakan PSO dan Backpropagation


## Deskripsi umum sistem
Algoritme PSO dan Backpropagation terletak di directory core. Algoritme ini hanya
sebagai class umum. Pada file main.py algoritme tersebut diimport dan fungsinya dikustomisasi dan diintegrasikan. Pada file main.py, fungsi yang dimodifikasi yaitu
- fungsi initWeight yang menggunakan bobot posisi partikel PSO
- fungsi set_fitness yang menjalankan algoritme backpropagation dengan bobot awal menggunakan bobot dari hasil algoritme PSO
- juga pada PSOxBackpro, yang inisialiasinya populasinya menggunakan partikel Backpropagation (BackpropagationParticle) yang sudah dikustomisasi sebelumnya