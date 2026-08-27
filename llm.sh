#!/bin/bash
# llm.sh – LLM System Docker Compose wrapper for AMD / NVIDIA / CPU
# Reads configuration from 'llm.ini' to manage startup parameters, port bindings, and GPU detection.

set -e

CONFIG_FILE="llm.ini"

# -------------------------------------------
# 1. Configuration Parser (Robust INI Reader)
# -------------------------------------------
parse_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "Error: $CONFIG_FILE not found." >&2
        exit 1
    fi

    # Initialize variables to defaults/empty
    SETUP_TYPE="8gb"
    BIND_PORT="default"
    LOCAL_ONLY=true
    HOSTS=""

    AMD_GPU_TARGET=""
    CUDA_ARCHITECTURES=""

    MIG_DEVICE_OLLAMA=""
    MIG_DEVICE_WHISPER=""

    ROOTLESS_MODE=false
    ACCOUNTS_MODE=false
    SKIP_DB_COPY=false

    local current_section=""

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Trim leading/trailing whitespace
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"

        # Skip comments and empty lines
        [[ -z "$line" || "$line" =~ ^[#\;] ]] && continue

        # Detect sections like [general]
        if [[ $line =~ ^\[(.+)\]$ ]]; then
            current_section="${BASH_REMATCH[1]}"
            continue
        fi

        # Parse key = value strictly
        if [[ "$line" =~ ^([a-zA-Z0-9_-]+)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local val="${BASH_REMATCH[2]}"

            # Remove surrounding quotes from value
            if [[ "$val" == \"*\" ]]; then
                val="${val#\"}"
                val="${val%\"}"
            fi

            case "${current_section}-${key}" in
            general-setup_type) SETUP_TYPE="$val" ;;

            networking-bind_port) BIND_PORT="$val" ;;
            networking-local_only)
                if [[ "$val" == "true" ]]; then LOCAL_ONLY=true; else LOCAL_ONLY=false; fi
                ;;
            networking-hosts) HOSTS="$val" ;;

            hardware_gpu-amd_gpu_target) AMD_GPU_TARGET="$val" ;;
            hardware_gpu-cuda_architectures) CUDA_ARCHITECTURES="$val" ;;

            nvidia_mig-mig_device_ollama) MIG_DEVICE_OLLAMA="$val" ;;
            nvidia_mig-mig_device_whisper) MIG_DEVICE_WHISPER="$val" ;;

            deployment-rootless_mode) [[ "$val" == "true" ]] && ROOTLESS_MODE=true ;;
            deployment-accounts_mode) [[ "$val" == "true" ]] && ACCOUNTS_MODE=true ;;
            deployment-skip_db_copy) [[ "$val" == "true" ]] && SKIP_DB_COPY=true ;;
            esac
        fi
    done <"$CONFIG_FILE"

    # ------------------------------------------------------------------
    # Build CORS_ALLOW_ORIGIN from HOSTS (if any) and export it
    # ------------------------------------------------------------------
    if [[ -n "$HOSTS" ]]; then
        IFS=';' read -ra HOST_ARRAY <<<"$HOSTS"
        ORIGINS=()
        for h in "${HOST_ARRAY[@]}"; do
            if [[ "$BIND_PORT" == "default" ]]; then
                ORIGINS+=("https://$h")
            else
                ORIGINS+=("https://$h:$BIND_PORT")
            fi
        done
        # Join with semicolons
        CORS_ALLOW_ORIGIN=$(
            IFS=';'
            echo "${ORIGINS[*]}"
        )
    else
        # No hosts specified -> allow all origins (behaviour for local-only setups)
        CORS_ALLOW_ORIGIN="*"
    fi

    # Export all needed variables (including HOSTS for the Caddyfile generator)
    export SETUP_TYPE OPEN_WEBUI_BIND=0.0.0.0:8080 HOSTS CORS_ALLOW_ORIGIN
    if [[ "$SKIP_DB_COPY" == "true" ]]; then export SKIP_DB_COPY=true; fi
}

# -------------------------------------------
# 2. Caddyfile Generator (Overwrites existing file)
# -------------------------------------------
generate_caddyfile() {
    echo "Generating Caddyfile: Local=${LOCAL_ONLY}, Port=${BIND_PORT}, Hosts='${HOSTS}'" >&2

    # Build host matcher strings if HOSTS is set
    local HOST_LIST_80=""
    local HOST_LIST_443=""
    local HOST_LIST_CUSTOM=""

    if [[ -n "$HOSTS" ]]; then
        IFS=';' read -ra HOST_ARRAY <<<"$HOSTS"
        if [[ "$BIND_PORT" == "default" ]]; then
            for h in "${HOST_ARRAY[@]}"; do
                HOST_LIST_80+="${h}:80, "
                HOST_LIST_443+="${h}:443, "
            done
            # Remove trailing ", "
            HOST_LIST_80="${HOST_LIST_80%, }"
            HOST_LIST_443="${HOST_LIST_443%, }"
        else
            for h in "${HOST_ARRAY[@]}"; do
                HOST_LIST_CUSTOM+="${h}:${BIND_PORT}, "
            done
            HOST_LIST_CUSTOM="${HOST_LIST_CUSTOM%, }"
        fi
    fi

    # Write Caddyfile header
    cat <<EOF >./caddy/Caddyfile
{
    auto_https off
}

EOF

    if [[ "$BIND_PORT" == "default" ]]; then
        # ---------- Port 80 (HTTP → HTTPS redirect) ----------
        if [[ -n "$HOST_LIST_80" ]]; then
            cat <<EOF >>./caddy/Caddyfile
${HOST_LIST_80} {
    redir https://{host}{uri}
}

EOF
        else
            cat <<EOF >>./caddy/Caddyfile
:80 {
    redir https://{host}{uri}
}

EOF
        fi

        # ---------- Port 443 (HTTPS) ----------
        if [[ -n "$HOST_LIST_443" ]]; then
            cat <<EOF >>./caddy/Caddyfile
${HOST_LIST_443} {
    tls /etc/caddy/server.crt /etc/caddy/server.key
    reverse_proxy llm-webui:8080 {
        header_up X-Forwarded-Proto {scheme}
    }
}
EOF
        else
            cat <<EOF >>./caddy/Caddyfile
:443 {
    tls /etc/caddy/server.crt /etc/caddy/server.key
    reverse_proxy llm-webui:8080 {
        header_up X-Forwarded-Proto {scheme}
    }
}
EOF
        fi
    else
        # ---------- Custom port (HTTPS only) ----------
        if [[ -n "$HOST_LIST_CUSTOM" ]]; then
            cat <<EOF >>./caddy/Caddyfile
${HOST_LIST_CUSTOM} {
    tls /etc/caddy/server.crt /etc/caddy/server.key
    reverse_proxy llm-webui:8080 {
        header_up X-Forwarded-Proto {scheme}
    }
}
EOF
        else
            cat <<EOF >>./caddy/Caddyfile
:${BIND_PORT} {
    tls /etc/caddy/server.crt /etc/caddy/server.key
    reverse_proxy llm-webui:8080 {
        header_up X-Forwarded-Proto {scheme}
    }
}
EOF
        fi
    fi
}

# -------------------------------------------
# 3. Dynamic Port Mapping Generator (Overwrites/creates overlay)
# -------------------------------------------
generate_ports_file() {
    # Determine IP binding prefix for the HOST side (empty string implies 0.0.0.0)
    local ADDR_PREFIX=""
    if [[ "$LOCAL_ONLY" == "true" ]]; then ADDR_PREFIX="127.0.0.1"; fi

    echo "Configuring Port Mapping: Local=${LOCAL_ONLY}, Bind=${BIND_PORT}" >&2

    if [[ "$BIND_PORT" == "default" ]]; then
        # Default mode requires exposing both 80 (HTTP->HTTPS) and 443 (HTTPS) on the host
        cat <<EOF >./caddy/caddy-ports.yaml
services:
  llm-caddy:
    ports:
      - "${ADDR_PREFIX}:80:80"
      - "${ADDR_PREFIX}:443:443"
EOF
    else
        # Custom port mode: expose only the specified port on host mapped to same port in container
        cat <<EOF >./caddy/caddy-ports.yaml
services:
  llm-caddy:
    ports:
      - "${ADDR_PREFIX}:${BIND_PORT}:${BIND_PORT}"
EOF
    fi
}

# -------------------------------------------
# 4. Hardware Detection & Helper Functions
# -------------------------------------------
detect_gpu_type() {
    if [ -e /dev/kfd ] && [ -e /dev/dri ]; then
        echo "amd"
        return 0
    fi
    if command -v nvidia-smi &>/dev/null; then
        echo "nvidia"
        return 0
    fi
    echo "cpu"
    return 0
}

detect_amd_gfx() {
    local GFX
    if command -v rocminfo &>/dev/null; then
        GFX=$(rocminfo | grep -oP 'Name:.*\bgfx[0-9a-f]+' | head -n1 | awk '{print $NF}')
        if [ -n "$GFX" ]; then
            case "$GFX" in gfx103[0-9]*)
                echo "gfx1030"
                return 0
                ;;
            gfx110[0-9]*)
                echo "gfx1100"
                return 0
                ;;
            *)
                echo "$GFX"
                return 0
                ;;
            esac
        fi
    fi
    if command -v rocm-smi &>/dev/null; then
        GFX=$(rocm-smi --showproductname 2>/dev/null | grep -oP 'gfx[0-9a-f]+' | head -n1)
        if [ -n "$GFX" ]; then
            case "$GFX" in gfx103[0-9]*)
                echo "gfx1030"
                return 0
                ;;
            gfx110[0-9]*)
                echo "gfx1100"
                return 0
                ;;
            *)
                echo "$GFX"
                return 0
                ;;
            esac
        fi
    fi
    echo "🔧 Warn: Could not determine AMD GPU Version. Using default gfx1030" >&2
    echo "gfx1030"
}

# -------------------------------------------
# Main Execution
# -------------------------------------------

parse_config

# 1. Generate Caddyfile (Overwrites ./caddy/Caddyfile)
generate_caddyfile

# 2. Generate Port Mapping Overlay (Overwrites/creates ./caddy/caddy-ports.yaml)
generate_ports_file

# 3. GPU Detection and Env Vars
GPU_TYPE=$(detect_gpu_type)
echo "Detected GPU type: $GPU_TYPE" >&2

if [[ "$GPU_TYPE" == "amd" ]]; then
    if [[ -z "$AMD_GPU_TARGET" || "$AMD_GPU_TARGET" == "" ]]; then AMD_GPU_TARGET=$(detect_amd_gfx); fi
    HSA_OVERRIDE="${AMD_GPU_TARGET/gfx/}"
    export AMDGPU_TARGETS="$AMD_GPU_TARGET"
    export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE%.*}.0"
fi

if [[ -n "$CUDA_ARCHITECTURES" ]]; then export CUDA_ARCHITECTURES; fi
if [[ -n "$MIG_DEVICE_OLLAMA" ]]; then export NVIDIA_VISIBLE_DEVICES_OLLAMA="$MIG_DEVICE_OLLAMA"; fi
if [[ -n "$MIG_DEVICE_WHISPER" ]]; then export NVIDIA_VISIBLE_DEVICES_WHISPER="$MIG_DEVICE_WHISPER"; fi

if [[ "$ACCOUNTS_MODE" == "true" ]]; then
    export WEBUI_AUTH=True
else
    export WEBUI_AUTH=False
fi
export WEBUI_SECRET_KEY="$(openssl rand -base64 32)"

# 4. Build and Execute Command
CMD=()
if [[ "$ROOTLESS_MODE" == "true" ]]; then
    CMD+=(docker compose -f docker-compose.yaml)
else
    ENV_VARS_TO_PASS=()
    ENV_VARS_TO_PASS+=("SETUP_TYPE=$SETUP_TYPE" "SKIP_DB_COPY=${SKIP_DB_COPY}")
    ENV_VARS_TO_PASS+=("WEBUI_AUTH=$WEBUI_AUTH")
    ENV_VARS_TO_PASS+=("CORS_ALLOW_ORIGIN=$CORS_ALLOW_ORIGIN")
    ENV_VARS_TO_PASS+=("WEBUI_SECRET_KEY=$WEBUI_SECRET_KEY")
    [[ -n "$AMDGPU_TARGETS" ]] && ENV_VARS_TO_PASS+=("AMDGPU_TARGETS=$AMDGPU_TARGETS" "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION")
    [[ -n "$CUDA_ARCHITECTURES" ]] && ENV_VARS_TO_PASS+=("CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES")
    [[ -n "$NVIDIA_VISIBLE_DEVICES_OLLAMA" ]] && ENV_VARS_TO_PASS+=("NVIDIA_VISIBLE_DEVICES_OLLAMA=$NVIDIA_VISIBLE_DEVICES_OLLAMA")
    [[ -n "$NVIDIA_VISIBLE_DEVICES_WHISPER" ]] && ENV_VARS_TO_PASS+=("NVIDIA_VISIBLE_DEVICES_WHISPER=$NVIDIA_VISIBLE_DEVICES_WHISPER")

    CMD+=(sudo)
    for var in "${ENV_VARS_TO_PASS[@]}"; do CMD+=("$var"); done
    CMD+=(docker compose -f docker-compose.yaml)
fi

case "$GPU_TYPE" in
amd) CMD+=(-f amd-gpu.yaml) ;;
nvidia)
    CMD+=(-f nvidia-gpu.yaml)
    if [[ -n "$NVIDIA_VISIBLE_DEVICES_OLLAMA" && -n "$NVIDIA_VISIBLE_DEVICES_WHISPER" ]]; then CMD+=(-f nvidia-gpu-mig.yaml); fi
    ;;
cpu) CMD+=(-f cpu.yaml) ;;
esac

# Add the dynamically generated ports overlay file
CMD+=(-f ./caddy/caddy-ports.yaml)

CMD+=("$@")

echo "Running:" >&2
echo "  ${CMD[*]}" >&2
exec "${CMD[@]}"
