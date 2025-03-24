# Vietnamese Poem Generator - Thất Ngôn Tứ Tuyệt

A web application for generating Vietnamese classical poems in the "Thất ngôn tứ tuyệt" style using machine learning models.

## Features

- Generate Vietnamese classical poems based on user prompts
- Choose between two different models:
  - Local model (using pre-trained weights)
  - Hugging Face model (via API)
- Modern, responsive UI built with Bootstrap 5
- FastAPI backend for efficient processing

## Setup Instructions

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. For the Hugging Face model, you'll need to:
   - Get an API key from Hugging Face
   - Update the `HUGGINGFACE_API_URL` in `app/main.py` with your specific model ID

3. Run the application:
   ```
   cd Poem_generation
   uvicorn app.main:app --reload
   ```

4. Open your browser and navigate to `http://localhost:8000`

## Project Structure

```
Poem_generation/
├── app/
│   ├── main.py              # FastAPI application
│   ├── static/              # Static assets
│   │   ├── css/             # CSS styles
│   │   └── js/              # JavaScript files
│   └── templates/           # HTML templates
├── model/                   # Local model weights
│   └── poem_generation_from_scratch_weights.pth
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## About Thất Ngôn Tứ Tuyệt

"Thất ngôn tứ tuyệt" is a traditional form of Vietnamese classical poetry that consists of four lines with seven words per line. This poetic form follows specific rules regarding tone patterns and rhyme schemes.

## License

This project is open source and available under the MIT License.