# Local AI (LLM) Deployment Stack

This project provides a fully-integrated local AI (LLM) deployment, ready to bring up with a simple `./llm.sh` command. It handles different modes of operation, such as rootless or rootful docker, AMD/NVIDIA/CPU hardware, professional NVIDIA MIG setups, binds to custom ports and redeployments with the same webui configuration (by copying the webui database).

Much of the project was coded, in turn, with help of AI. However, code was always reviewed and diffed by a human before applying it to the codebase. Should any commits / pull requests to this project arise, it is expected that any AI-generated code is reviewed with similar scrutiny.


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
