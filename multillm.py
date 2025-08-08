from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
import re
from sentence_transformers import SentenceTransformer, util
from openai import AzureOpenAI
import requests

endpointGPT = "https://muham-mbdq7l3l-eastus2.cognitiveservices.azure.com/"
endpoint = "https://muham-mbdq7l3l-eastus2.services.ai.azure.com/models"

subscription_key = "DBnEvQeAbpcjfx0MEznGvtm8EIkccvyvrVDrArgzaM3PJY1Dp55EJQQJ99BFACHYHv6XJ3w3AAAAACOGFt40"

api_versionGPT = "2024-12-01-preview"
api_version = "2024-05-01-preview"

model1 = "gpt-4.1"
model2 = "DeepSeek-R1"
model3 = "MAI-DS-R1"

tingkatPendidikan = "SMP"
kontenKebudayaan = ["Angklung", "Gerantang", "Sampe", "Geundrang", "Kujang", "Rencong", "Dohong", "Keris Bali",
                    "Surabi", "Kue Bhoi", "Lemang", "Ayam Betutu", "Julang Ngapak", "Rangkang", "Lamin", "Bale Dauh"]

systemPrompt = f"Anda adalah asisten yang tepat, mudah digunakan, dan mandiri. Gunakan gaya bahasa yang mudah dipahami untuk tingkat pendidikan {tingkatPendidikan}. Ubah gaya bahasa dan sesuaikan tingkatan taksonomi bloom untuk tingkat Pendidikan {tingkatPendidikan} sebelum memberikan jawaban akhir. Anda adalah seorang guru budaya Indonesia."

modelEmbedding = SentenceTransformer('all-MiniLM-L6-v2')

clientGPT = AzureOpenAI(
    api_version=api_versionGPT,
    azure_endpoint=endpointGPT,
    api_key=subscription_key,
)

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(
        "DBnEvQeAbpcjfx0MEznGvtm8EIkccvyvrVDrArgzaM3PJY1Dp55EJQQJ99BFACHYHv6XJ3w3AAAAACOGFt40"),
    api_version=api_version,
)


def remove_think_content(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def clean_response(text):
    # Hapus tag <think>...</think>
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Hapus pembuka seperti '''json atau ```json
    text = re.sub(r"^[`']{3}json\s*", "", text,
                  flags=re.IGNORECASE | re.MULTILINE)
    # Hapus penutup triple quote (jika ada)
    text = re.sub(r"[`']{3}\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def getResponse(model, prompt):
    if model == model1:
        response = clientGPT.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": systemPrompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_completion_tokens=800,
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=model
        )
        return response.choices[0].message.content
    else:
        response = client.complete(
            messages=[
                SystemMessage(content=systemPrompt),
                UserMessage(content=prompt),
            ],
            max_tokens=2048,
            model=model2
        )
        return clean_response(remove_think_content(response.choices[0].message.content))


def GetResponseMultiLLM(deskripsi, pertanyaan, tingkatPendidikan, konten):
    promptPertanyaan = f'''Anda adalah seorang ahli kebudayaan Indonesia sekaligus guru untuk tingkat {tingkatPendidikan}. Tugas anda adalah menjawab pertanyaan berikut:
  {pertanyaan}

  Jika pertanyaan di luar konteks {konten}, berikan respon kepada pengguna untuk mencari tau informasi tersebut di luar game pembelajaran ini! 

  Buat jawaban tersebut agar menyesuaikan tingkat taksonomi bloom dan gaya bahasa untuk {tingkatPendidikan}. 

  Berikan jawaban dalam bahasa Indonesia dan bentuk format JSON seperti berikut:

  {{
    'Jawaban': 'tuliskan jawaban anda di sini',
  }}
  '''

    promptLLMAAJ = f'''Anda adalah seorang ahli kebudyaan Indonesia sekaligus guru untuk tingkat {tingkatPendidikan}, dan LLM yang bertindak sebagai penentu keputusan. Tugas anda adalah menilai respon dari 3 model LLM dan menentukan jawaban terbaik untuk pertanyaan dalam konteks deskripsi konten kebudayaan berikut:
  {deskripsi}

  Berikut adalah respon dari 3 model LLM tersebut atas pertanyaan {pertanyaan}:
  LLM 1: {getResponse(model1, promptPertanyaan)}
  LLM 2: {getResponse(model2, promptPertanyaan)}
  LLM 3: {getResponse(model3, promptPertanyaan)}

  Buat jawaban akhir agar menyesuaikan tingkat taksonomi bloom dan gaya bahasa untuk {tingkatPendidikan}. 

  Berikan jawaban dalam bahasa Indonesia dan bentuk format JSON seperti berikut:
  {{
    'jawaban': 'jawaban akhir dari pertanyaan',
  }}
  '''

    response = getResponse(model1, promptLLMAAJ)
    return response


def CalculateCosineSimilarity(response, ground_truth):
    response_embeddings = modelEmbedding.encode(
        response, convert_to_tensor=True)
    ground_truth_embeddings = modelEmbedding.encode(
        ground_truth, convert_to_tensor=True)
    cosine_similarities = util.cos_sim(
        response_embeddings, ground_truth_embeddings).diagonal()
    return float(cosine_similarities)


def search_web_serper(query):
    """Function search_web_serper yang diperbaiki"""
    api_key = "376506b5c5d13039ed93fd20b35d73acfa307427"
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    # PERBAIKAN 1: Bersihkan query dari format JSON jika ada
    original_query = query
    if isinstance(query, str) and query.startswith('{'):
        try:
            import json
            query_obj = json.loads(query)
            if 'query' in query_obj:
                query = query_obj['query']
        except json.JSONDecodeError:
            pass

    # PERBAIKAN 2: Bersihkan query dari karakter tidak perlu
    query = str(query).strip('"\'`{}').strip()

    # PERBAIKAN 3: Tambahkan parameter geo dan language untuk hasil yang lebih relevan
    payload = {
        "q": query,
        "num": 5,      # Tingkatkan dari 3 ke 5
        "gl": "id",    # Geographic location: Indonesia
        "hl": "id"     # Host language: Indonesian
    }

    try:
        # PERBAIKAN 4: Tambahkan timeout yang lebih panjang
        response = requests.post(url, headers=headers,
                                 json=payload, timeout=15)

        # PERBAIKAN 5: Handle berbagai status code
        if response.status_code == 200:
            results = response.json()
            organic_results = results.get("organic", [])

            # PERBAIKAN 6: Log jika hasil kosong (opsional, bisa dihapus untuk production)
            if not organic_results:
                print(
                    f"Warning: Tidak ada hasil organik untuk query: '{query}'")
                # Cek apakah ada data di key lain
                if 'answerBox' in results:
                    print(
                        "Ditemukan answerBox, mungkin bisa digunakan sebagai alternatif")
                if 'peopleAlsoAsk' in results:
                    print(
                        "Ditemukan peopleAlsoAsk, mungkin bisa digunakan sebagai alternatif")

            return organic_results

        elif response.status_code == 401:
            print("Error: API Key tidak valid atau expired")
            return []
        elif response.status_code == 429:
            print("Error: Rate limit exceeded, coba lagi nanti")
            return []
        else:
            print(f"Error {response.status_code}: {response.text}")
            return []

    except requests.exceptions.Timeout:
        print("Error: Request timeout")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return []


def summarize_evidence(search_results):
    snippets = "\n".join([r.get("snippet", "") for r in search_results])
    prompt = f"""
    Anda adalah asisten untuk menyarikan informasi. Berikut adalah beberapa hasil pencarian:
    {snippets}

    Berikan ringkasan fakta-fakta penting dari hasil pencarian ini dalam bentuk ringkasan objektif:
    """
    return getResponse(model1, prompt)


def reflect_on_evidence(question, answer, summary):
    prompt = f"""
    Pertanyaan: {question}
    Jawaban kandidat: {answer}
    Ringkasan bukti: {summary}

    Apakah bukti ini mendukung, bertentangan, atau tidak cukup untuk mengevaluasi jawaban tersebut? Jelaskan secara singkat.
    """
    return getResponse(model1, prompt)


def evaluate_with_tale(question, answer, max_iterations=3):
    memory = []

    for i in range(max_iterations):
        print(f"\nIterasi {i + 1} dari {max_iterations}")
        last_reflection = memory[-1]["reflection"] if i != 0 else None
        query_prompt = f"""Pertanyaan: {question}
            Jawaban kandidat: {answer}
            Refleksi sebelumnya: {last_reflection}
            Buat query pencarian web baru untuk memverifikasi jawaban ini.
            Jawab menggunakan format JSON seperti ini:
            {{
                "query": "[tuliskan query pencarian anda di sini]"
            }}"""
        query = getResponse(model1, query_prompt)
        print("Query: " + query)

        search_results = search_web_serper(query)
        print("Hasil pencarian: ", search_results)
        evidence_summary = summarize_evidence(search_results)
        print("Ringkasan bukti: " + evidence_summary)
        reflection = reflect_on_evidence(question, answer, evidence_summary)
        print("Refleksi: " + reflection)

        memory.append({
            "query": query,
            "evidence": evidence_summary,
            "reflection": reflection
        })

    # Buat penilaian akhir
    combined_evidence = "\n".join([m["evidence"] for m in memory])
    combined_reflections = "\n".join([m["reflection"] for m in memory])

    judgment_prompt = f"""
    Pertanyaan: {question}
    Jawaban kandidat: {answer}

    Berikut adalah ringkasan bukti yang ditemukan:
    {combined_evidence}

    Berikut adalah refleksi terhadap bukti tersebut:
    {combined_reflections}

    Berdasarkan informasi ini, apakah jawaban kandidat BENAR atau SALAH? Berikan keputusan dalam format JSON seperti ini:
    {{
      "verdict": 1,  # 1 jika benar, 0 jika salah
      "rationale": "Penjelasan..."
    }}
    """
    final_judgment = getResponse(model1, judgment_prompt)
    print("Keputusan akhir: " + final_judgment)
    return final_judgment


def __main__():
    # tingkatPendidikan = "SMP"
    # konten = "Angklung"
    # kontenKebudayaan = ["Angklung", "Gerantang", "Sampe", "Geundrang", "Kujang", "Rencong", "Dohong", "Keris Bali",
    #                     "Surabi", "Kue Bhoi", "Lemang", "Ayam Betutu", "Julang Ngapak", "Rangkang", "Lamin", "Bale Dauh"]
    # deskripsi = "Angklung adalah alat musik tradisional Indonesia yang terbuat dari bambu. Gerantang adalah alat musik tradisional yang dimainkan dengan cara dipukul. Sampe adalah alat musik petik tradisional dari Kalimantan. Geundrang adalah alat musik tradisional Aceh yang terbuat dari kayu. Kujang adalah senjata tradisional Sunda yang juga dianggap sebagai simbol budaya. Rencong adalah senjata tradisional Aceh yang memiliki bentuk unik. Dohong adalah alat musik tiup tradisional dari Papua. Keris Bali adalah senjata tradisional Bali yang memiliki nilai spiritual tinggi. Surabi adalah kue tradisional Indonesia yang terbuat dari tepung beras. Kue Bhoi adalah kue khas Betawi yang terbuat dari tepung ketan. Lemang adalah makanan khas Indonesia yang terbuat dari beras ketan dan dimasak dalam bambu. Ayam Betutu adalah hidangan khas Bali yang terkenal pedas. Julang Ngapak adalah burung endemik Indonesia yang menjadi simbol budaya Bali. Rangkang adalah alat musik tradisional dari Nusa Tenggara Timur. Lamin adalah rumah adat suku Dayak di Kalimantan. Bale Dauh adalah bangunan tradisional Bali yang digunakan untuk upacara keagamaan."
    # pertanyaan = "Siapa penemu angklung?"

    # promptPertanyaan = f'''Anda adalah seorang ahli kebudayaan Indonesia sekaligus guru untuk tingkat {tingkatPendidikan}. Tugas anda adalah menjawab pertanyaan berikut:
    # {pertanyaan}

    # Jika pertanyaan di luar konteks {konten}, berikan respon kepada pengguna untuk mencari tau informasi tersebut di luar game pembelajaran ini!

    # Buat jawaban tersebut agar menyesuaikan tingkat taksonomi bloom dan gaya bahasa untuk {tingkatPendidikan}.

    # Berikan jawaban dalam bahasa Indonesia dan bentuk format JSON seperti berikut:

    # {{
    #   'Jawaban': 'tuliskan jawaban anda di sini',
    # }}
    # '''

    # response = getResponse(model1, promptPertanyaan)
    # cosineSimilarity = CalculateCosineSimilarity(
    #     response, "Angklung adalah alat musik tradisional Indonesia yang terbuat dari bambu.")
    # if cosineSimilarity < 0.7:
    #     response = GetResponseMultiLLM(
    #         deskripsi, pertanyaan, tingkatPendidikan, kontenKebudayaan)
    #     print(f"Cosine Similarity: {cosineSimilarity}")
    #     print("Jawaban akhir dari LLM AAJ:")
    #     print(response)
    #     hasil_tale = evaluate_with_tale(pertanyaan, response)
    #     print("Hasil evaluasi TALE:")
    #     print(hasil_tale)
    # else:
    #     print(f"Cosine Similarity: {cosineSimilarity}")
    #     print(response)

    pertanyaan = "Apakah terdapat alat musik lain yang menyerupai angklung di negara lain?"
    jawaban = "Ya, di negara lain juga ada alat musik yang mirip dengan angklung, terutama cara memainkannya yang digoyang. Misalnya, di Filipina ada alat musik bambu yang disebut bungkaka atau bamboo rattle. Di Amerika Latin, ada alat musik maracas, yaitu alat dari labu yang diisi biji-bijian dan digoyang untuk menghasilkan suara. Di Brasil, ada caxixi, yaitu keranjang kecil yang diisi kerikil dan digoyang juga. Walaupun prinsip dasarnya sama, yaitu bunyi muncul karena digoyang, angklung tetap punya keunikan sendiri. Angklung terbuat dari bambu, dimainkan dengan cara digoyang, dan masing-masing angklung bisa menghasilkan satu nada, sehingga bisa dimainkan bersama-sama membentuk musik. Jadi, meski ada alat musik mirip di negara lain, angklung tetap khas Indonesia, terutama dari budaya Sunda di Jawa Barat."
    hasil_tale = evaluate_with_tale(pertanyaan, jawaban)


if __name__ == "__main__":
    __main__()


# def search_web_serper_debug(query):
#     api_key = "376506b5c5d13039ed93fd20b35d73acfa307427"
#     url = "https://google.serper.dev/search"
#     headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

#     # Debug: Pastikan query adalah string, bukan JSON
#     if isinstance(query, str) and query.startswith('{'):
#         try:
#             query_obj = json.loads(query)
#             if 'query' in query_obj:
#                 query = query_obj['query']
#         except json.JSONDecodeError:
#             pass

#     payload = {"q": query, "num": 3}

#     print(f"=== DEBUG SERPER API ===")
#     print(f"URL: {url}")
#     print(f"Query: {query}")
#     print(f"Payload: {payload}")

#     try:
#         response = requests.post(url, headers=headers,
#                                  json=payload, timeout=10)
#         print(f"Status Code: {response.status_code}")
#         print(f"Response Headers: {dict(response.headers)}")

#         if response.status_code == 200:
#             results = response.json()
#             print(
#                 f"Full Response: {json.dumps(results, indent=2, ensure_ascii=False)}")

#             # Cek berbagai kemungkinan struktur response
#             organic_results = results.get("organic", [])
#             if not organic_results:
#                 print("WARNING: Organic results kosong!")
#                 print("Available keys in response:", list(results.keys()))

#                 # Cek apakah ada results di key lain
#                 for key in ['results', 'searchInformation', 'answerBox']:
#                     if key in results:
#                         print(f"Found data in '{key}': {results[key]}")

#             return organic_results
#         else:
#             print(f"Error: {response.status_code}")
#             print(f"Response Text: {response.text}")
#             return []

#     except requests.exceptions.RequestException as e:
#         print(f"Request Error: {e}")
#         return []


# def test_api_key():
#     """Test sederhana untuk memverifikasi API key"""
#     test_query = "python programming"
#     results = search_web_serper_debug(test_query)

#     if results:
#         print("✅ API key working!")
#         print(f"Found {len(results)} results")
#     else:
#         print("❌ API key tidak bekerja atau ada masalah lain")

# # Perbaikan untuk function evaluate_with_tale


# def evaluate_with_tale_fixed(question, answer, max_iterations=3):
#     memory = []

#     for i in range(max_iterations):
#         print(f"\nIterasi {i + 1} dari {max_iterations}")
#         last_reflection = memory[-1]["reflection"] if memory else None

#         query_prompt = f"""Pertanyaan: {question}
#         Jawaban kandidat: {answer}
#         Refleksi sebelumnya: {last_reflection}

#         Buat query pencarian web yang SINGKAT dan SPESIFIK untuk memverifikasi jawaban ini.
#         Jawab HANYA dengan kata kunci pencarian, jangan dalam format JSON.

#         Contoh yang baik: "angklung alat musik bambu"
#         Contoh yang buruk: "Apakah angklung benar-benar alat musik tradisional?"
#         """

#         query_response = getResponse(model1, query_prompt)

#         # Ekstrak query yang bersih
#         query = query_response.strip()

#         # Jika masih dalam format JSON, ekstrak
#         if query.startswith('{'):
#             try:
#                 query_obj = json.loads(query)
#                 query = query_obj.get('query', query)
#             except:
#                 pass

#         # Bersihkan query dari tanda kutip dan karakter tidak perlu
#         query = query.strip('"\'`{}')

#         print(f"Query yang akan digunakan: '{query}'")

#         search_results = search_web_serper_debug(query)

#         if not search_results:
#             print("Tidak ada hasil pencarian, mencoba query alternatif...")
#             # Fallback query yang lebih sederhana
#             simple_query = question.split()[:3]  # Ambil 3 kata pertama
#             simple_query = " ".join(simple_query)
#             search_results = search_web_serper_debug(simple_query)

#         if search_results:
#             evidence_summary = summarize_evidence(search_results)
#             print("Ringkasan bukti: " + evidence_summary)
#             reflection = reflect_on_evidence(
#                 question, answer, evidence_summary)
#             print("Refleksi: " + reflection)

#             memory.append({
#                 "query": query,
#                 "evidence": evidence_summary,
#                 "reflection": reflection
#             })
#         else:
#             print("Tidak dapat menemukan hasil pencarian yang valid")
#             break

#     # Lanjutkan dengan penilaian akhir jika ada memory
#     if memory:
#         combined_evidence = "\n".join([m["evidence"] for m in memory])
#         combined_reflections = "\n".join([m["reflection"] for m in memory])

#         judgment_prompt = f"""
#         Pertanyaan: {question}
#         Jawaban kandidat: {answer}

#         Berikut adalah ringkasan bukti yang ditemukan:
#         {combined_evidence}

#         Berikut adalah refleksi terhadap bukti tersebut:
#         {combined_reflections}

#         Berdasarkan informasi ini, apakah jawaban kandidat BENAR atau SALAH?
#         Berikan keputusan dalam format JSON seperti ini:
#         {{
#           "verdict": 1,
#           "rationale": "Penjelasan..."
#         }}
#         """
#         final_judgment = getResponse(model1, judgment_prompt)
#         print("Keputusan akhir: " + final_judgment)
#         return final_judgment
#     else:
#         return '{"verdict": 0, "rationale": "Tidak dapat memverifikasi karena tidak ada hasil pencarian"}'


# # Test the API
# if __name__ == "__main__":
#     test_api_key()
