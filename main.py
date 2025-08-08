# api_server.py
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
import requests
from pydantic import BaseModel
from multillm import getResponse, GetResponseMultiLLM, evaluate_with_tale, CalculateCosineSimilarity
import io
from fastapi.responses import JSONResponse

app = FastAPI()

groundTruth = '''Angklung adalah alat musik multitonal (bernada ganda) yang secara tradisional berkembang dalam masyarakat berbahasa Sunda di Pulau Jawa bagian barat. Alat musik ini dibuat dari bambu, dibunyikan dengan cara digoyangkan (bunyi disebabkan oleh benturan badan pipa bambu) sehingga menghasilkan bunyi yang bergetar dalam susunan nada 2, 3, sampai 4 nada dalam setiap ukuran, baik besar maupun kecil. Angklung terdaftar sebagai Karya Agung Warisan Budaya Lisan dan Nonbendawi Manusia dari UNESCO sejak November 2010. 

Gerantang memiliki susunan atas beberapa potongan bambu yang tersusun berderet dan digunakan dengan 2 alat pemukul khusus seperti contoh Gambang (alat musik dari Jawa) tetapi alat musik Gerantang memakai bambu. Alat musik tradisional ini asalnya dari Bali cukup sering dipakai pada kegiatan gamelan atau angklung. Pada daerah Jawa Barat alat ini disebut calung, jelas pastinya terdapat perbedaan antara alat musik tradisional Bali dan Jawa. Alat musik Gerantang digunakan pada pentas seni Cupak Gerantang.
Cupak Gerantang merupakan cerita 2 orang tokoh kakak beradik yang mempunyai nama Cupak dan Grantang. Yang semuanya memiliki sifatnya masing-masing seperti Cupak mencerminkan semua sifat buruk manusia, sedangkan Gerantang kebalikannya, ia mencerminkan sifat baik di diri manusia. Untuk membuat alat musik tradisional ini, diperlukan bambu, gergaji, parang, amplas (penghalus) dan beberapa benda tambahan seperti contoh obeng atau palu, Bahan bambu untuk gerantang sekitar 1 sampai 3 ruas dan mempunyai panjang 45 ~ 95 cm. Pada bilah bambu tersebut berilah lubang, mungkin berkisar seperempat bagian bambu sebagai tempat untuk mendapatkan suara yang diinginkan. Panjang bungbung gerantang sekitar sati ruas sampai dengan tiga ruas, atau kurang antara 45 cm sampai 95 cm dari nada tertinggi sampai nada rendahnya. Alat yang perlu disiapkan sebagai pembuat adalah gergaji sebagai pemotong , parang untuk menebas, dan pengutik sebagai penghalusnya. Gerantang terbuat dari bambu. Instrumen ini terdiri atas beberapa potongan bambu yang disusun berderet dan dimainkan dengan menggunakan 2 alat pemukul khusus, mirip seperti gambang (alat musik dari Jawa). Gerantang cukup sering digunakan dalam pementasan gamelan atau pertunjukan angklung. Di daerah Jawa barat, instrumen seperti ini disebut calung. Selain itu, instrumen gerantang juga digunakan dalam pentas seni Cupak Gerantang. Sebuah kesenian yang menceritakan dua tokoh kakak beradik bernama Cupak dan Gerantang. Cupak mencerminkan sifat buruk manusia, sedangkan Gerantang sebaliknya. Detail Spesifikasi gerantang Panjang bambu untuk membuat gerantang sekitar satu sampai tiga ruas (kira-kira 45 cm hingga 95 cm). Terdapat lubang di sekitar seperempat bagian bambu untuk menghasilkan suara. Panjang bungbung gerantang berkisar antara satu ruas sampai dengan tiga ruas, atau antara 45 cm sampai 95 cm dari nada tertinggi hingga terendah.

Sampe adalah alat musik tradisional Suku Dayak Kenyah. Alat musik ini terbuat dari kayu adau. Namun, yang paling sering dijadikan bahan adalah kayu arrow, kayu kapur, dan kayu ulin dan dibuat secara tradisional. Proses pembuatan bisa memakan waktu berminggu minggu. Dibuat dengan 3 senar melody, bass, dan minor. Biasanya sampe terdapat ukiran pada ujungnya, ukiran tersebut saling bersambungan yang melambangkan persaudaraan dan kesatuan tak terputus diantara suku Dayak Kenyah. Akan diukir sesuai dengan keinginan pembuatnya, kerap ditampilkan  atau dimainkan pada acara  kebesaran bahkan pada upacara kematian.

Geundrang merupakan salah satu unit alat musik tradisional Aceh yang merupakan bagian dari perangkatan musik Serune Kalee. Geundrang termasuk jenis alat musik yang dibunyikan dengan cara dipukul baik dengan menggunakan tangan atau memakai kayu pemukul. Geundrang dijumpai di daerah Aceh Besar dan juga dijumpai di daerah pesisir Aceh seperti Pidie dan Aceh Utara. Fungsi Geundrang nerupakan alat pelengkap tempo dari musik tradisional etnik Aceh.

Dalam Wacana dan Khasanah Kebudayaan Nusantara, Kujang diakui sebagai senjata tradisional masyarakat Masyarakat Jawa Barat (Sunda) dan Kujang dikenal sebagai senjata yang memiliki nilai sakral serta mempunyai kekuatan magis. Beberapa peneliti menyatakan bahwa istilah Kujang berasal dari kata Kudihyang dengan akar kata Kudi dan Hyang. Kudi diambil dari bahasa Sunda Kuno yang artinya senjata yang mempunyai kekuatan gaib sakti, sebagai jimat, sebagai penolak bala, misalnya untuk menghalau musuh atau menghindari bahaya/penyakit. Senjata ini juga disimpan sebagai pusaka, yang digunakan untuk melindungi rumah dari bahaya dengan meletakkannya di dalam sebuah peti atau tempat tertentu di dalam rumah atau dengan meletakkannya di atas tempat tidur (Hazeu, 1904 : 405-406) Sedangkan Hyang dapat disejajarkan dengan pengertian Dewa dalam beberapa mitologi, namun bagi masyarakat Sunda Hyang mempunyai arti dan kedudukan di atas Dewa. Secara umum, Kujang mempunyai pengertian sebagai pusaka yang mempunyai kekuatan tertentu yang berasal dari para dewa (=Hyang), dan sebagai sebuah senjata, sejak dahulu hingga saat ini Kujang menempati satu posisi yang sangat khusus di kalangan masyarakat Jawa Barat (Sunda). Sebagai lambang atau simbol dengan niali-nilai filosofis yang terkandung di dalamnya, Kujang dipakai sebagai salah satu estetika dalam beberapa lambang organisasi serta pemerintahan. Disamping itu, Kujang pun dipakai pula sebagai sebuah nama dari berbagai organisasi, kesatuan dan tentunya dipakai pula oleh Pemda Propinsi Jawa Barat. Di masa lalu Kujang tidak dapat dipisahkan dari kehidupan masyarakat Sunda karena fungsinya sebagai peralatan pertanian. Pernyataan ini tertera dalam naskah kuno Sanghyang Siksa Kanda Ng Karesian (1518 M) maupun tradisi lisan yang berkembang di beberapa daerah diantaranya di daerah Rancah, Ciamis. Bukti yang memperkuat pernyataan bahwa kujang sebagai peralatan berladang masih dapat kita saksikan hingga saat ini pada masyarakat Baduy, Banten dan Pancer Pangawinan di Sukabumi. Dengan perkembangan kemajuan, teknologi, budaya, sosial dan ekonomi masyarakat Sunda, Kujang pun mengalami perkembangan dan pergeseran bentuk, fungsi dan makna. Dari sebuah peralatan pertanian, kujang berkembang menjadi sebuah benda yang memiliki karakter tersendiri dan cenderung menjadi senjata yang bernilai simbolik dan sakral. Wujud baru kujang tersebut seperti yang kita kenal saat ini diperkirakan lahir antara abad 9 sampai abad 12.

Rencong adalah senjata tradisional Aceh, bentuknya menyerupai huruf L, dan bila dilihat lebih dekat bentuknya merupakan kaligrafi tulisan bismillah. Rencong termasuk dalam kategori dagger atau belati (bukan pisau ataupun pedang). Rencong memiliki kemiripan rupa dengan keris. Panjang mata pisau rencong dapat bervariasi dari 10 cm sampai 50 cm. Matau pisau tersebut dapat berlengkung seperti keris, namun dalam banyak rencong, dapat juga lurus seperti pedang. Rencong dimasukkan ke dalam sarung belati yang terbuat dari kayu, gading, tanduk, atau kadang-kadang logam perak atau emas. Dalam pembawaan, rencong diselipkan di antara sabuk di depan perut pemakai. Rencong memiliki tingkatan; untuk raja atau sultan biasanya sarungnya terbuat dari gading dan mata pisaunya dari emas dan berukirkan sekutip ayat suci dari Alquran agama Islam. Sedangkan rencong-rencong lainnya biasanya terbuat dari tanduk kerbau ataupun kayu sebagai sarungnya, dan kuningan atau besi putih sebagai belatinya. Seperti kepercayaan keris dalam masyarakat Jawa, masyarakat tradisional Aceh menghubungkan kekuatan mistik dengan senjata rencong. Rencong masih digunakan dan dipakai sebagai atribut busana dalam upacara tradisional Aceh. Masyarakat Aceh mempercayai bahwa bentuk dari rencong mewakili simbol dari basmalah dari kepercayaan agama Islam. Rencong begitu populer di masyarakat Aceh sehingga Aceh juga dikenal dengan sebutan "Tanah Rencong".

Dohong, Kalimantan Barat nama dari senjata tradisional masyarakat Kalimantan Barat  khususnya masyarakat Dayak Ngaju adalah Dohong. Dohong mempunyai bentuk sebuah mata tombak yang bisa juga berfungsi sebagai  pisau. Senjata ini dipercaya sebagai senjata yang tertua. Jika dulunya senjata ini di gunakan untuk berperang, namun saat ini senjata Dohong di gunakan sebagai alat untuk memotong tali pusar dan juga di gunakan untuk menyembelih hewan qurban. Saat ini, yang bisa memiliki senjata Dohong ini hanyalah ketua dari suku dasar sendiri atau seorang pisur.

Keris bali berasal dari suku Bali. Keris Bali sama seperti keris pada umunya, namun bedanya adalah lebih besar dan panjang. Balutan sarung keris berpenampilan mewah, dihias dengan emas, perak, gading, dan bahannya kayu langka yang dihiasi ukiran.

Serabi Kadang disebut srabi atau surabi merupakan salah satu makanan ringan atau jajanan pasar yang berasal dari Indonesia. Serabi serupa dengan pancake (pannekoek atau pannenkoek) namun terbuat dari tepung beras (bukan tepung terigu) dan diberi kuah cair yang manis (biasanya dari gula kelapa). Kuah ini bervariasi menurut daerah di Indonesia. Daerah yang terkenal dengan kue serabinya adalah Jakarta, Bandung, Solo, Pekalongan dan Purwokerto yang masing-masing memiliki keunikan tersendiri. Ada juga surabi Arab yang terkenal karena keunikannya yang terdapat di kota bogor.

Kue Bhoi adalah penganan khas Aceh Besar yang dikenal luas oleh masyarakat Aceh. Bentuk kue ini sangat bervariasi, seperti : bentuk ikan, bintang, bunga, dan lain-lain. Kue Bhoi ini dapat menjadikan salah satu buah tangan ketika akan berkunjung ke sanak saudara atau tetangga yang mengadakan hajatan atau pesta, seperti sunatan dan kelahiran. Kue Bhoi ini mempunyai harga yang sangat relatif murah, satu kemasan berkisar dengan harga Rp. 5.000,- ,10.000,- bahkan ada yang ratusan ribu. Kue Bhoi juga dijadikan sebagai salah satu isi dari bingkisan seserahan yang dibawa oleh calon pengantin pria untuk calon pengantin perempuan pada saat acara pernikahan. Kue Bhoi sendiri biasanya diperoleh di pasar-pasar tradisional ataupun dipesan langsung pada pembuatnya. Proses pembuatan kue Bhoi ini pun tergolong sedikit rumit. Pasalnya, tidak semua orang bisa membuat kuliner ini dan dibutuhkan kesabaran serta keuletan.

Lemang merupakan salah satu makanan yang selalu ada dalam acara adat masyarakat Dayak, misalnya pada ritual Lom Plai. Untuk cara membuatnya, kira-kira bisa dijabarkan sebagai berikut: Beras ketan yang direndam selama satu malam dimasukkan ke dalam potongan bambu beralas daun pisang di dalamnya. Setelah itu, santan yang telah dibubuhi sedikit garam dituang ke dalam potongan bambu. Lemang kemudian dimasak dengan cara memanggangnya di atas bara api.

Ayam Betutu adalah lauk yang terbuat dari ayam yang utuh yang berisi bumbu, kemudian dipanggang dalam api sekam. Betutu ini telah dikenal di seluruh kabupaten di Bali. Salah satu produsen betutu adalah desa Melinggih, kecamatam payangan kabupaten Gianyar. Ayam betutu juga merupakan makanan khas Gilimanuk. Betutu digunakan sebagai sajian pada upacara keagamaan dan upacara adat serta sebagai hidangan dan di jual. Konsumennya tidak hanya masyarakat Bali tapi juga tamu manca negara yang datang ke Bali, khususnya pada tempat-tempat tertentu seperti di hotel dan rumah makan/restoran. Betutu tidak tahan disimpan lama.

Julang Ngapak sendiri adalah salah satu nama dan model dari rumah adat sunda. Selain Julan Ngapak, rumah adat Sunda sebenarnya memiliki nama lainnya seperti Buka Pongpok, Capit Gunting, Jubleg Nangkub, Badak Heuay, Tagog Anjing, dan Perahu Kemureb, dimana kesemuanya berbentuk rumah panggung dan hal yang membedakan satu dengan lainnya adalah bentuk atap dan pintu rumahnya. Julang Ngapak sendiri mempunyai arti burung yang merentangkan sayap, jadi bentuk atap pada bangunan ini bentuknya melebar pada kedua sisinya tetapi berbeda antara sudut atas dan sudut bawahnya...kita akan seperti melihat burung julak yang sedang merentangkan kedua sayapnya jika melihat bangunan ini, apalagi jika kita melihatnya dari samping. Sayangnya, kesemua bentuk rumah adat Sunda ini sudah semakin sulit ditemui karena masyarakat yang mempertahankan rumahnya dalam bentuk rumah adat sudah semakin jarang dan yang masih mempertahankannya hanya bisa ditemui di pelosok-pelosok daerah saja.  Di kabupaten sumedang, konsep rumah adat Julang Ngapak ini salah satunya bisa dilihat pada bangunan Kantor Bupati Sumedang seperti terlihat di atas, hanya saja bentuknya tidak berupa rumah panggung. Bentuk rumah adat Julang Ngapak ini diambil menjadi konsep bangunan Kantor Bupati sebagai pengejawantahan nilai-nilai budaya Sunda di Kabupaten Sumedang, juga dalam rangka mempertegas jati diri Sunda dalam Lingkungan Puseur Budaya Sunda.

Rangkang berupa rumah panggung yang hanya terdiri dari satu ruangan yang berasal dari Aceh. Rangkang ini biasanya dimanfaatkan sebagai tempat melepas lelah bagi petani saat sedang bertani. Material yang digunakan untuk membuat Rangkang juga sangat sederhana yaitu kayu biasa dan daun rumbia untuk atapnya.

Dalam bahasa Dayak Benuaq, lamin berarti Lou, disebut rumah panjang karena ada yang mencapai puluhan meter hingga ratusan meter panjangnya. Lamin dibuat berdasarkan kebutuhan penghuninya. Semakin banyak penghuni lamin, maka akan semakin panjang lamin itu dibuat. Lamin di Pondok Labu memiliki panjang sekitar 36 meter dan lebar 10 meter. Lamin utama memiliki panjang sekitar 25 meter. Tinggi lantai lamin dari permukaan tanah 2 meter, dengan jumlah kamar 6 kamar tidur. Dilengkapi dengan 2 dapur dan 2 kamar mandi yang terletak dikiri kanan lamin. Lamin ini bangunan asli yang dibangun orang Dayak untuk digunakan sebagai tempat tinggal, tempat berkumpul, dan tempat melakukan aktifitas komunitasnya (Haris Sukendar dkk, 2006:94-95). Didepan lamin terdapat 8 tiang belontak yang merupakan tiang berukir yang dipergunakan sebagai tempat mengikat hewan yang akan dipersembahkan dalam upacara adat, seperti kerbau. Didepan lamin juga terdapat balai untuk meletakkan sesaji, tiang rumah untuk menopang lamin ini ada 12 tiang, yang ditempatkan berjajar enam tiang dibagian depan dan enam tiang dibelakang, yang terbuat dari kayu ulin dengan diameter 20cm. pada bagian atas tiang rumah yang terletak dibagian depan lamin masing-masing terdapat 2 tanduk kerbau, yang merupakan kerbau persembahan upacara adat. Dalam mitologinya kerbau terlahir dari perkawinan terlarang antara Serupui Petah dan saudara kandung ayahnya. Oleh karena itu, kerbau dikorbankan agar kestabilan alam dan lingkungan tetap terjaga dari pengaruh buruk akibat perkawinan salah tersebut (Riwut, 2003:204). Terdapat 3 tangga yang digunakan, bagian tengah tangga utama (tukar dusun) terbuat dari papan kayu dengan 8 anak tangga, berukuran lebar 1meter dan jarak 20cm. 2 tangga lainnya berada di kiri kanan tangga utama, yang terbuat dari kayu ulin utuh, yang dipahat membentuk anak tangga. Pada bagian kanan berjumlah 9 anak tangga dan bagian kiri berjumlah 10 anak tangga, bagian atas tangga ini dibentuk seperti kepala manusia. Bagian depan lamin terdapat 3 pintu masuk dan 4 jendela. Pintu utama berhadapan dengan tangga utama, dengan ukuran tinggi 2m dan lebar 90cm. Sedang 2 lainnya terletak dikanan kiri lamin yang berhadapan dengan tangga yang lainnya, dengan ukuran yang lebih sempit, tinggi 2m dan lebar 80cm. Ukuran keempat jendala sama besar yaitu tinggi 80cm dan lebar 60cm. Adanya perubahan orientasi arah hadap hunian sehingga lamin tidak dibangun mengikuti arah sungai.

Bale Dauh yang digunakan oleh masyarakat Bali sebagai tempat khusus untuk menerima tamu. Ruangan ini juga difungsikan sebagai tempat tidur bagi anak remaja laki-laki. Bale Dauh bentuknya sama dengan Bale Manten, yaitu persegi panjang. Akan tetapi, letaknya berada di bagian dalam ruangan, tidak di sudut. Untuk posisinya sendiri di sisi barat dan lantainya harus lebih rendah dibanding Bale Manten. Lalu ciri khas lainnya, tiang penyangga di Bale Dauh ini jumlahnya berbeda antara satu rumah dengan rumah lainnya.
'''


class PromptRequest(BaseModel):
    deskripsi: str
    pertanyaan: str
    tingkat_pendidikan: str
    konten: str


@app.post("/ask")
async def process_prompt(req: PromptRequest):
    promptPertanyaan = f'''Anda adalah seorang ahli kebudayaan Indonesia sekaligus guru untuk tingkat {req.tingkat_pendidikan}. Tugas anda adalah menjawab pertanyaan berikut:
    {req.pertanyaan}

    Jika pertanyaan di luar konteks {req.konten}, berikan respon kepada pengguna untuk mencari tau informasi tersebut di luar game pembelajaran ini! 

    Buat jawaban tersebut agar menyesuaikan tingkat taksonomi bloom dan gaya bahasa untuk {req.tingkat_pendidikan}. 

    Berikan jawaban dalam bahasa Indonesia dan bentuk format JSON seperti berikut:

    {{
    'Jawaban': 'tuliskan jawaban anda di sini',
    }}'''

    response = getResponse("gpt-4.1", promptPertanyaan)
    cosine = CalculateCosineSimilarity(response, groundTruth)

    if cosine < 0.7:
        response = GetResponseMultiLLM(
            req.deskripsi, req.pertanyaan, req.tingkat_pendidikan, req.konten)
        hasil_tale = evaluate_with_tale(req.pertanyaan, response)

        # Bersihkan hasil_tale dari string JSON mentah
        import json
        try:
            parsed_tale = json.loads(hasil_tale)
        except json.JSONDecodeError:
            parsed_tale = {"verdict": 0,
                           "rationale": "Format hasil_tale tidak valid"}

        if parsed_tale.get("verdict", 0) == 0:
            fallback_prompt = f"""
            Anda adalah seorang guru kebudayaan Indonesia untuk tingkat {req.tingkat_pendidikan}.
            Setelah mengevaluasi jawaban terhadap pertanyaan berikut:

            "{req.pertanyaan}"

            Ditemukan bahwa tidak ada informasi yang memadai untuk menjawab pertanyaan tersebut secara akurat.
            Berikan jawaban sopan yang menyampaikan kepada pengguna bahwa pertanyaan tersebut tidak dapat dijawab saat ini,
            dan arahkan mereka untuk mencari informasi di luar game pembelajaran ini.

            Jawaban dalam format JSON:
            {{
            "Jawaban": "..."
            }}
            """

            response = getResponse("gpt-4.1", fallback_prompt)

    else:
        hasil_tale = {
            "verdict": 1,
            "rationale": "Jawaban relevan dengan deskripsi konten"
        }

    return {
        "cosine_similarity": cosine,
        "response": response,
        "hasil_tale": hasil_tale
    }

API_KEY = "AJ8DKCl86QeMmoVTums8fbgiy7DEhtZpatS0ANDmbesRCPAAhiztJQQJ99BGACHYHv6XJ3w3AAAAACOGbZXs"
ENDPOINT = "https://muhammadfarizh31-2599-resource.cognitiveservices.azure.com/"
DEPLOYMENT_NAME = "gpt-4o-mini-transcribe"
API_VERSION = "2024-03-01-preview"


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp3", ".wav")):
        raise HTTPException(
            status_code=400, detail="File harus format .mp3 atau .wav")

    # Kirim audio ke Azure Whisper Endpoint
    url = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/audio/transcriptions?api-version={API_VERSION}"
    headers = {
        "api-key": API_KEY,
        "language": "id",  # Bahasa Indonesia
    }

    # Baca file dan buat multipart/form-data
    file_bytes = await file.read()
    files = {
        "file": (file.filename, file_bytes, file.content_type),
        "model": (None, "whisper-1"),  # wajib "whisper-1" meski GPT-4o mini
        "language": (None, "id"),  # Bahasa Indonesia
    }

    response = requests.post(url, headers=headers, files=files)

    # Respon
    if response.status_code == 200:
        result = response.json()
        return {"transkripsi": result["text"]}
    else:
        raise HTTPException(
            status_code=response.status_code, detail=response.text)
