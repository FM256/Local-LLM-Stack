# Local AI (LLM) Deployment Stack

This project provides a fully-integrated local AI (LLM) deployment, ready to bring up with a simple `./llm.sh` command. It handles different modes of operation, such as rootless or rootful docker, AMD/NVIDIA/CPU hardware, professional NVIDIA MIG setups, binds to custom ports and redeployments with the same webui configuration (by copying the webui database).

## Integrated / Configured Features

* Inference using ollama models (automatically downloaded based on your configured hardware profile)
* User-friendly UI with persistent & temporary chats, overviews, advanced parameter settings, custom
  model, skills and tools definitions, etc. (thanks to Open WebUI)
* Text to Speech & Speech to Text support; Voice mode for chatting hands-free with the LLM
* Flexible deployment with support for rootful and rootless docker, custom port bindings, database persistence across redeployments and simple autologin vs account-based authentication
* Secure Deployment with CORS policy, HTTPs by default and restricted CA management
* Support for both AMD and NVIDIA GPUs, including NVIDIA MIG support for professional GPU partitioning
* Fully configured RAG pipeline with embedding (based on mxbai-embed-large in ollama) and reranking
  (based on a custom reranking container).

## Major Components

* Open WebUI: The Frontend
* Ollama: The LLM Engine
* Whisper: The speech-to-text engine (GPU accelerated)
* Kokoro: The text-to-speech engine
* Reranker: A lightweight custom reranker container that runs quickly, running just on CPU. Proved to be more restart-friendly than other alternatives, and did not show any performance drawbacks when tested.
* Setup: A small custom container that downloads ollama models and redeploys a previously exported webui database
* Caddy: The reverse-proxy, providing SSL termination for secure HTTPS connections
* Tools: A separate tools container. Currently contains an endpoint to transcribe audio and video files, as well as youtube video URLs

The RAG embedding is done by the `mxbai-embed-large` model through `ollama`.

## Initial Setup

### Installing Dependencies

You need the following dependencies:
* `docker`
* `docker-compose`
* Working GPU drivers for your system
    - AMD rocm
    - NVIDIA Container Toolkit

### Generating a Certificate

Since this project expects HTTPs-based communication, you must first generate an SSL certificate by
some means.

For a local, self-signed deployment, you should use the `caddy/create-restricted-ca.sh` script for
these purposes. The script will generate a restricted CA certificate (only able to issue
certificates for your specified hosts), which greatly limits the worry of subsequently installing
the CA certificate into the browser.

Ensure to list all the hosts you would like your service to be reachable through, e.g. if you want it to be reachable as `https://www.myhost.com:3000` you will need to
specify `www.myhost.com`. To reach it as `https://192.168.0.5` you must add `192.168.0.5`. For example: `cd caddy && ./create-restricted-ca.sh myhost myhost.local`

Leave the generated `server.crt` and `server.key` inside the `caddy` directory, and install the `ca.crt` into the certificates store of your browser (under `Authorities`), so that it trusts the certificate when you connect.

### Configuring the Service

Copy the configuration template for modification: `cp llm.ini.template llm.ini`

The template documents the meaning of each configuration option. Again, ensure to specify your hosts
correctly. Otherwise, if you connect to your service through an unspecified hostname URL, either the
certificate check will fail (review CA installation and the certificate hosts), or it succeeds but
you are met with a blank page (review `hosts =` in your `llm.ini`).

### Launching the Service

With everything configured, you can simply run `./llm.sh up` to launch the service. You can provide
any docker compose command, such as `./llm.sh down` or `./llm.sh restart llm-caddy`.

The first launch will take a while to build containers and finally launch open-webui. If you face
internal server error, the Open WebUI might not be fully launched yet. Once you see the Open WebUI
banner in your log output, you should be able to successfully connect through the browser.

## Contributing / AI Policy

This project was developed with AI assistance, with reviewing, diffing and validation done by humans. In case you want to contribute in a way that hinges on AI support, it is expected that you fully validate your code before submission.
