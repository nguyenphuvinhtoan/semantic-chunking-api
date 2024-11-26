## 1. Clone the repo:

```bash
git clone https://github.com/nguyenphuvinhtoan/semantic-chunking-api.git
```

## 2. Create a virtual environment using conda:

```bash
conda create -p ./.env python=3.11 -y
conda activate ./.env
pip install -r requirements.txt
pip install -e langchain-experimental/libs/experimental/.
```

## 3. Run the server:

```bash
flask run --host 0.0.0.0 --port=8001 --debug
```

## 4. Test with API:

Field in form-data:

- document: <upload file transcript>
- sentences_per_group: <number of sentences want to group> (Optional)

Example cURL:

```bash
curl --location 'http://localhost:8001/chunk' \
--form 'document=@"postman-cloud:///1efabc8f-8d1e-4990-9c99-7ba0127da772"' \
--form 'sentences_per_group="2"'
```
