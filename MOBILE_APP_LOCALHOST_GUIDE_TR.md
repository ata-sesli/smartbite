# SmartBite Mobil Entegrasyon Rehberi

Bu proje mobil uygulama icin **Secenek B - HTTP sunucu** seklinde calisir. Mobil uygulamanin icine `label_scanner.tflite` gomulmuyor, model cloud'a deploy edilmiyor. Gelistirme ve test senaryosunda backend, projeyi calistiran bilgisayarda Docker ile acilir; telefon da ayni yerel agdan bu backend'e istek atar.

Kisa karar cumlesi:

> Bu repo icin mobil entegrasyon yolu HTTP'dir: uygulama `POST /mobile/expiry-scans` ile fotograf yukler, gerekiyorsa `PATCH /mobile/expiry-scans/{id}` ile kullanici duzeltmesini geri yollar.

## Repo acilinca ilk okunacak ozet

Backend su parcaciklardan olusur:

- `api`: HTTP endpoint'lerini acar. Varsayilan port `8005`.
- `worker`: kuyruklu scan isleri icin arka plan iscisidir.
- `postgres`: veritabani.
- `redis`: kuyruk altyapisi.
- `models/`: yerel model dosyalari. Cloud model servisi yok.
- `data/storage`: yuklenen dosyalar ve runtime ciktilari icin yerel storage.

Mobil uygulama tarafinda esas alinacak endpoint'ler:

- `GET /health`: backend ayakta mi kontrolu.
- `POST /mobile/expiry-scans`: telefondan cekilen fotografi multipart olarak yukle, son kullanma tarihi analizi al.
- `PATCH /mobile/expiry-scans/{id}`: kullanici tarihi elle duzeltirse backend'e kaydet.

Bu endpoint su anda `productName` uretmez ve SKT/TETT ayrimi yapmaz. Mobil uygulamada bu alanlar gerekiyorsa simdilik:

- `productName`: bos string veya `null`
- `expiryDate`: response icindeki `expiry_date`
- `dateType`: `"unknown"`
- `confidence`: response icindeki `recognition_confidence` veya `detector_confidence`
- `rawOcrText`: response icindeki `raw_text`

Yani arkadasinin ajaninin bekledigi `productName`, `expiryDate`, `dateType` sabit JSON kontrati bu repoda birebir yok. Dogru kontrat asagidaki `POST /mobile/expiry-scans` kontratidir.

## Gereksinimler

Bilgisayarda sunlar kurulu olmali:

- Docker Desktop veya OrbStack
- Git

Telefonla test icin:

- Telefon ve bilgisayar ayni Wi-Fi aginda olmali.
- Bilgisayarin firewall'u `8005` portuna yerel agdan gelen isteklere izin vermeli.
- Mobil uygulamada base URL olarak `localhost` degil, bilgisayarin yerel IP adresi kullanilmali.

## Backend'i Docker ile calistirma

Repo kok dizininde:

```bash
cp .env.example .env
docker compose up -d --build
```

Bu komutlar `postgres`, `redis`, `api` ve `worker` servislerini baslatir. API container'i acilirken Alembic migration'lari otomatik calistirir.

Servisleri kontrol etmek icin:

```bash
docker compose ps
```

Loglari izlemek icin:

```bash
docker compose logs -f api
docker compose logs -f worker
```

Backend ayakta mi kontrolu:

```bash
curl -sS http://localhost:8005/health
```

Beklenen cevap buna benzer:

```json
{
  "status": "ok",
  "app": "smartbite",
  "timestamp": "2026-05-13T12:00:00Z"
}
```

Servisleri kapatmak icin:

```bash
docker compose down
```

## Mobil endpoint 1: fotograf yukleme ve tarih analizi

Endpoint:

```text
POST /mobile/expiry-scans
Content-Type: multipart/form-data
```

Base URL bilgisayardan test ederken:

```text
http://localhost:8005
```

Gercek telefondan test ederken:

```text
http://BILGISAYARIN_YEREL_IP_ADRESI:8005
```

Form alanlari:

| Alan | Zorunlu mu? | Aciklama |
| --- | --- | --- |
| `image` | Evet | JPEG, PNG veya WEBP fotograf. Maksimum 10 MB. |
| `message` | Hayir | Mobil uygulamadan kisa not/string gondermek istenirse. |
| `metadata` | Hayir | JSON object string'i. Ornek: `{"platform":"ios","source":"dev"}` |

Ornek `curl`:

```bash
curl -sS -X POST http://localhost:8005/mobile/expiry-scans \
  -F "image=@test64/IMG_0885.JPG;type=image/jpeg" \
  -F 'metadata={"source":"mobile-dev","platform":"ios"}'
```

Basarili cevap ornegi:

```json
{
  "id": "2d8b5c5d-0d14-49d2-9d8f-99c3d31f1a2b",
  "status": "done",
  "expiry_date": "2026-05-31",
  "detected_expiry_date": "2026-05-31",
  "corrected_expiry_date": null,
  "raw_text": "31.05.2026",
  "normalized_text": "31.05.2026",
  "recognition_confidence": 0.92,
  "detector_confidence": 0.74,
  "reason": "selected best parsed candidate",
  "created_at": "2026-05-13T12:00:00Z"
}
```

Tarih bulunamazsa `expiry_date` ve `detected_expiry_date` `null` gelebilir. Bu durumda mobil uygulama kullaniciya manuel tarih secimi gostermeli ve secilen tarihi ikinci endpoint ile backend'e gondermelidir.

Mobil uygulama icin onerilen response mapping:

```json
{
  "productName": null,
  "expiryDate": "<expiry_date>",
  "dateType": "unknown",
  "confidence": "<recognition_confidence>",
  "rawOcrText": "<raw_text>"
}
```

Not: Backend'in asil response'unu degistirmeden uygulama icinde bu mapping yapilabilir. Eger mobil uygulama mutlaka `productName`, `expiryDate`, `dateType` alanlarini backend'den birebir bekleyecekse, backend'e ayrica adapter endpoint eklemek gerekir.

## Mobil endpoint 2: kullanici duzeltmesini kaydetme

Endpoint:

```text
PATCH /mobile/expiry-scans/{id}
Content-Type: application/json
```

Body:

```json
{
  "corrected_expiry_date": "2026-05-31",
  "reason": "mobile user correction"
}
```

Ornek `curl`:

```bash
curl -sS -X PATCH http://localhost:8005/mobile/expiry-scans/2d8b5c5d-0d14-49d2-9d8f-99c3d31f1a2b \
  -H "Content-Type: application/json" \
  -d '{"corrected_expiry_date":"2026-05-31","reason":"mobile user correction"}'
```

Basarili cevap yine scan kaydini dondurur. Bu kez:

- `corrected_expiry_date` kullanicinin sectigi tarih olur.
- `expiry_date`, varsa `corrected_expiry_date` degerini kullanir.

## Gercek telefondan localhost testi

Telefonun `localhost` adresi telefonun kendisidir; bilgisayardaki backend degildir. Bu yuzden telefondan istek atarken bilgisayarin yerel IP adresi kullanilir.

Mac'te yerel IP adresini bulmak icin:

```bash
ipconfig getifaddr en0
```

Eger bos donerse:

```bash
ipconfig getifaddr en1
```

Ornek IP `192.168.1.42` ise mobil uygulamadaki base URL:

```text
http://192.168.1.42:8005
```

Telefondaki tarayicidan once sunu ac:

```text
http://192.168.1.42:8005/health
```

Health cevabi gorunuyorsa mobil uygulama da ayni base URL ile calismalidir.

### Android notlari

Fiziksel Android telefonda:

```text
http://192.168.1.42:8005
```

Android Emulator kullaniliyorsa bilgisayar host'una ulasmak icin genelde:

```text
http://10.0.2.2:8005
```

Android uygulamada HTTP cleartext kapaliysa development icin cleartext izni gerekebilir. Production gibi dusunmeyin; bu lokal test icindir.

### iOS notlari

Fiziksel iPhone'da:

```text
http://192.168.1.42:8005
```

iOS Simulator'da bilgisayarin localhost'una genelde su adresle gidilebilir:

```text
http://localhost:8005
```

iOS uygulamada HTTP kullanimi App Transport Security tarafindan engellenirse development icin ilgili domain/IP icin gecici ATS exception gerekir. Production deploy yapmiyoruz; bu lokal gelistirme ayaridir.

## Sik gorulen hatalar

### Telefonda `localhost:8005` calismiyor

Normal. Telefonda `localhost`, telefonun kendisini gosterir. Bilgisayarin Wi-Fi IP adresini kullan:

```text
http://192.168.x.x:8005
```

### `/health` bilgisayarda calisiyor ama telefonda calismiyor

Kontrol listesi:

- Telefon ve bilgisayar ayni Wi-Fi'da mi?
- VPN, hotspot izolasyonu veya kurumsal ag cihazlari local cihazlar arasi trafigi engelliyor mu?
- Firewall `8005` portunu engelliyor mu?
- Base URL `http://` ile mi basliyor?
- IP adresi dogru mu?

### Upload `400 unsupported content type` donuyor

`image` alaninin content type'i su degerlerden biri olmali:

- `image/jpeg`
- `image/png`
- `image/webp`

Ayrica dosya uzantisi da `.jpg`, `.jpeg`, `.png` veya `.webp` olmali.

### Upload `400 file too large` donuyor

Maksimum upload boyutu varsayilan olarak 10 MB. Mobil uygulama fotografi gondermeden once kaliteyi veya boyutu dusurebilir.

### Ilk istek yavas geliyor

Normal olabilir. Model/pipeline ilk kez yuklenirken gecikme olur. Sonraki istekler daha hizli olur.

### `docker compose up -d --build` uzun suruyor

Ilk build AI bagimliliklarini kurdugu icin uzun surebilir. Sonraki build'ler cache kullandigi icin daha hizli olur.

## Mobil ajan icin net uygulama talimati

1. Backend'i uygulama icine gomulu model gibi dusunme; `.tflite` bekleme.
2. Base URL'yi environment/config ile degistirilebilir yap.
3. Lokal bilgisayarda test ederken base URL `http://localhost:8005`.
4. Fiziksel telefonda test ederken base URL `http://<bilgisayarin-lan-ip-adresi>:8005`.
5. Fotograf yukleme icin multipart alan adi kesinlikle `image` olsun.
6. Basarili upload cevabindaki `id` degerini sakla; kullanici duzeltme yaparsa `PATCH /mobile/expiry-scans/{id}` icin bu id gerekir.
7. Backend su anda urun adi uretmiyor. UI urun adi istiyorsa kullanicidan alin veya bos goster.
8. Backend su anda SKT/TETT ayrimi uretmiyor. Uygulama tarafinda `dateType` degerini simdilik `"unknown"` kabul et.
9. Tarih alani icin backend'in canonical alani `expiry_date`; format `YYYY-MM-DD`.
10. `expiry_date` `null` ise kullaniciya manuel tarih secme akisini ac.

## Minimum mobil pseudo-code

```ts
const baseUrl = "http://192.168.1.42:8005";

async function uploadExpiryImage(file: Blob) {
  const form = new FormData();
  form.append("image", file, "scan.jpg");
  form.append("metadata", JSON.stringify({ source: "mobile-dev" }));

  const response = await fetch(`${baseUrl}/mobile/expiry-scans`, {
    method: "POST",
    body: form
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  const scan = await response.json();

  return {
    id: scan.id,
    productName: null,
    expiryDate: scan.expiry_date,
    dateType: "unknown",
    confidence: scan.recognition_confidence ?? scan.detector_confidence,
    rawOcrText: scan.raw_text,
    backendRaw: scan
  };
}

async function correctExpiryDate(scanId: string, correctedDate: string) {
  const response = await fetch(`${baseUrl}/mobile/expiry-scans/${scanId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      corrected_expiry_date: correctedDate,
      reason: "mobile user correction"
    })
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}
```

