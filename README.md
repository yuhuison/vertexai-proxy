# Vertex AI to OpenAI Proxy

An OpenAI-compatible API proxy for Google Gemini and Anthropic Claude on Vertex AI. **One deployment solves all the headaches.**

## Why This Project?

Google's official API uses the Gemini API format, not the OpenAI format. While Google does offer an OpenAI-compatible endpoint, it comes with significant pain points:

- **Safety Filter Bypass is Broken**: Gemini has built-in NSFW content filtering. The native Gemini API lets you disable it directly, but the OpenAI-compatible endpoint requires you to pass safety settings via `extra_body` — which many clients and SDKs don't support well.
- **Aggressive Rate Limiting**: The official OpenAI-compatible endpoint is extremely prone to `429 Too Many Requests` errors.
- **`thought_signature` Nightmare**: Gemini 3.0+ models return an encrypted `thought_signature` field during multi-turn tool calling that must be passed back in subsequent requests. This is completely non-standard and breaks most OpenAI clients out of the box.
- **Auth Fragmentation**: Vertex AI and AI Studio use entirely different authentication mechanisms. Switching between them requires code changes.

**This project handles all of the above for you.** Deploy once on Cloud Run, and use it as a standard OpenAI-compatible API endpoint with any client (NextChat, LobeChat, Cursor, etc.) — no hacks, no `extra_body`, no special handling required.

> [!TIP]
> **Recommended: Deploy on Google Cloud Run (Free Tier)**
>
> Google Cloud Run offers a [generous free tier](https://cloud.google.com/run/pricing) (2 million requests/month, 360,000 GB-seconds of memory, etc.). Since this proxy is a lightweight, stateless application, **it will cost you essentially nothing for personal use**. Additionally, deploying on Cloud Run means **you don't need to deal with Vertex AI's complex authentication at all** — Cloud Run's service account handles it automatically via ADC. This is the recommended deployment method.

## Features

- **Full OpenAI API Compatibility**: Drop-in replacement for `/v1/chat/completions`.
- **Streaming, Tools, Structured Output, Multimodal**: Full support for SSE streaming, Function Calling, JSON Schema outputs, and image inputs.
- **Gemini 3.0+ `thought_signature` Handled**: Automatically extracts, caches, and passes `thought_signature` so multi-turn tool calling just works.
- **Vertex AI ADC Auth**: Uses Application Default Credentials on Cloud Run — no API keys to manage for Google services.

> [!CAUTION]
> **You MUST set your own `MASTER_KEY` environment variable before deploying.**
>
> The `MASTER_KEY` is used to authenticate incoming API requests to your proxy. If you do not set a custom value, a **hardcoded default key** will be used, which means **anyone who knows this default key can freely use your proxy and consume your Google Cloud credits**.
>
> Set it via the `MASTER_KEY` environment variable during deployment (see Step 4 below). **The author of this project assumes no responsibility for any financial loss, unauthorized usage, or security incidents caused by failing to change the default key. Use at your own risk.**

## Getting Started

### 1. Fork the Repository

To get started, fork this repository to your own GitHub account:
1. Click the **Fork** button at the top right of this page.
2. Select your account as the destination.
3. Clone your forked repository to your local machine:
   ```bash
   git clone https://github.com/YOUR_USERNAME/vertexai-proxy.git
   cd vertexai-proxy
   ```

### 2. Google Cloud Setup

You need a Google Cloud Project with the **Vertex AI API** enabled.

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing one.
3. Search for **Vertex AI API** in the top search bar and click **Enable**.

### 3. Service Account & Permissions (Keyless Access)

To run this application securely without hardcoding service account JSON files or API keys, we will assign a Service Account to Google Cloud Run and grant it the necessary Vertex AI permissions. Cloud Run will automatically use these Application Default Credentials (ADC).

1. You can use the default Compute Engine service account (`PROJECT_NUMBER-compute@developer.gserviceaccount.com`) or create a new dedicated service account in the **IAM & Admin** dashboard.
2. Grant the **Vertex AI User** (`roles/aiplatform.user`) role to this service account so it has permission to invoke models.

Using the `gcloud` CLI:
```bash
# Set your project ID
export PROJECT_ID="your-google-cloud-project-id"
# Set your service account email
export SERVICE_ACCOUNT="your-service-account@$PROJECT_ID.iam.gserviceaccount.com"

# Grant the Vertex AI User role
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/aiplatform.user"
```

### 4. Deploy to Google Cloud Run

You can deploy the proxy directly from the source code to Cloud Run. The proxy uses the `GOOGLE_CLOUD_PROJECT` and `GOOGLE_GENAI_USE_VERTEXAI` environment variables to route requests securely through Vertex AI without needing Google API keys.

#### Method A: Using Google Cloud CLI (gcloud)

Run the following command from the root of your cloned repository:

```bash
gcloud run deploy vertex-ai-proxy \
  --source . \
  --region us-central1 \
  --service-account $SERVICE_ACCOUNT \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_GENAI_USE_VERTEXAI=true,MASTER_KEY=your-custom-secret-key"
```

#### Method B: Using Google Cloud Console (No CLI Required)

If you don't have the `gcloud` CLI installed, you can deploy directly from your GitHub repository using the Google Cloud Console UI:

1. Go to [Cloud Run](https://console.cloud.google.com/run) in the Google Cloud Console.
2. Click **Create Service** and select **Continuously deploy from a repository**.
3. Connect your GitHub account and select your forked `vertexai-proxy` repository.
4. Under **Authentication**, select **Allow unauthenticated invocations**.
5. Expand the **Container(s), Volumes, Networking, Security** section at the bottom.
6. In the **Security** tab, find the **Service account** dropdown and select the service account you configured in Step 3.
7. Click **Create** to deploy.

*(Note: Cloud Run will automatically pick up your Project ID and Vertex AI configuration from the Service Account. If you want to use a custom API key instead of the default to protect your proxy, you can set the `MASTER_KEY` environment variable in the **Variables & Secrets** tab).*

*(Note: If you have already deployed and just need to assign the service account, click **Edit & Deploy New Revision**, go to the **Security** tab, select your service account, and click **Deploy**).*

**Environment Variables Reference:**
- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID (Required).
- `GOOGLE_GENAI_USE_VERTEXAI`: Set to `true` to use Vertex AI ADC without an API key.
- `MASTER_KEY`: Your custom secret key to protect *your* proxy. OpenAI clients will use this as their Bearer token.
- `GOOGLE_CLOUD_LOCATION`: (Optional) Region for Gemini models, defaults to `global`.
- `CLAUDE_LOCATION`: (Optional) Region for Claude models, defaults to `global` (Note: some Claude models may require specific regions like `us-east5`).

### Optional: Firestore for Multi-Turn Tool Calling

> [!NOTE]
> **This step is only required if you use Gemini 3.0+ models with multi-turn tool calling** (i.e., the model calls a tool, you send the result back, and the model calls another tool). If you only use the proxy for regular AI chat, you can skip this entirely.

Gemini 3.0+ models return an encrypted `thought_signature` during tool calls that must be echoed back in subsequent requests. This proxy caches them in Firestore so it works seamlessly across Cloud Run's stateless instances.

**Setup:**
1. Go to [Firestore](https://console.cloud.google.com/firestore) in the Google Cloud Console.
2. Create a **new database** (not the default one) with the ID `thought-signature-cache`.
3. Choose **Native mode** and any region.
4. Grant your Cloud Run service account the **Cloud Datastore User** (`roles/datastore.user`) role:
   ```bash
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:$SERVICE_ACCOUNT" \
     --role="roles/datastore.user"
   ```

That's it. The proxy will automatically detect and use the Firestore database. If Firestore is not configured, the proxy still works normally — only multi-turn tool calling with `thought_signature` will be affected.

### 5. Usage

Once deployed, Cloud Run will provide you with a public URL (e.g., `https://vertex-ai-proxy-something.a.run.app`).

You can now use this URL as your `base_url` in any OpenAI-compatible client, SDK, or UI (like NextChat, LobeChat, etc.).

**Example Request (cURL):**
```bash
curl -X POST https://your-cloud-run-url.a.run.app/v1/chat/completions \
  -H "Authorization: Bearer your-custom-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3-flash-preview",
    "messages": [{"role": "user", "content": "Hello! How can you assist me today?"}]
  }'
```

## Adding New Models

To add support for new models, update the mappings in `config.py`:
- For Vertex AI Gemini models: Update `GEMINI_MODEL_MAPPING`
- For Vertex AI Claude models: Update `CLAUDE_MODEL_MAPPING`

## Alternative: Deploy Without GCP (Docker)

If you prefer not to use Google Cloud Run, you can run the proxy anywhere using Docker. In this case, you will need to provide a **Google Cloud Service Account Key JSON** file for authentication.

```bash
docker build -t vertex-ai-proxy .

docker run -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT=your-project-id \
  -e GOOGLE_GENAI_USE_VERTEXAI=true \
  -e MASTER_KEY=your-custom-secret-key \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  -v /path/to/your/service-account-key.json:/app/credentials.json:ro \
  vertex-ai-proxy
```

> [!NOTE]
> When running outside of GCP, Application Default Credentials are not automatically available. You must download a service account key JSON from `IAM & Admin > Service Accounts > Keys` in the Google Cloud Console and mount it into the container via the `GOOGLE_APPLICATION_CREDENTIALS` environment variable as shown above.
