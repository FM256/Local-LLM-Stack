#!/bin/bash
# create-restricted-ca.sh
# Creates a Root CA and server certificate with SAN entries for all given domains and IPs.
# The Root CA is restricted by Name Constraints to only issue for those exact names/IPs.
# Usage: ./create-restricted-ca.sh hostname1 [hostname2 ...] [IP1 ...] [IP2 ...]
# Example: ./create-restricted-ca.sh myhost nas.local 192.168.1.100 10.0.0.5

set -e

if [ $# -eq 0 ]; then
    echo "ERROR: No arguments provided."
    echo "Usage: $0 <name1> [name2 ...]"
    echo "  Each argument can be a domain name (e.g., myhost.local) or an IP address (IPv4 or IPv6)."
    exit 1
fi

DOMAINS=("$@")
VALIDITY_DAYS="3650"

echo "Creating restricted Root CA for: ${DOMAINS[*]}"

# --------------------------------------------
# 1. Build SAN and Name Constraints strings
# --------------------------------------------
# Always include localhost and loopback IPs
SAN_STR="DNS:localhost, IP:127.0.0.1, IP:::1"
NAMECONSTRAINT_STR=""

# Helper to add a DNS entry
add_dns() {
    local d="$1"
    SAN_STR="${SAN_STR}, DNS:$d"
    if [ -n "$NAMECONSTRAINT_STR" ]; then NAMECONSTRAINT_STR="${NAMECONSTRAINT_STR}, "; fi
    NAMECONSTRAINT_STR="${NAMECONSTRAINT_STR}permitted;DNS:$d, permitted;DNS:.$d"
}

# Helper to add an IP entry (with required mask for nameConstraints)
add_ip() {
    local ip="$1"
    SAN_STR="${SAN_STR}, IP:$ip"
    # Determine the correct mask for the IP version
    local mask=""
    if [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        mask="255.255.255.255" # IPv4 exact match
    else
        mask="ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff" # IPv6 exact match
    fi
    if [ -n "$NAMECONSTRAINT_STR" ]; then NAMECONSTRAINT_STR="${NAMECONSTRAINT_STR}, "; fi
    NAMECONSTRAINT_STR="${NAMECONSTRAINT_STR}permitted;IP:$ip/$mask"
}

for arg in "${DOMAINS[@]}"; do
    # Basic IPv4 detection (dotted quad)
    if [[ "$arg" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        add_ip "$arg"
    # Basic IPv6 detection (contains colon)
    elif [[ "$arg" =~ : ]]; then
        add_ip "$arg"
    else
        add_dns "$arg"
    fi
done

echo "SAN entries: $SAN_STR"
echo "Name Constraints: $NAMECONSTRAINT_STR"

# --------------------------------------------
# 2. Generate Root CA private key
# --------------------------------------------
echo "Generating Root CA key..."
openssl genrsa -out ca.key 2048

# --------------------------------------------
# 3. Create config for Root CA (self-signed)
# --------------------------------------------
cat >ca.cnf <<EOF
[ req ]
default_bits       = 2048
distinguished_name = req_distinguished_name
x509_extensions    = v3_ca
prompt             = no

[ req_distinguished_name ]
CN = Restricted CA for local use

[ v3_ca ]
basicConstraints = critical, CA:TRUE
keyUsage = critical, keyCertSign, cRLSign
nameConstraints = critical, $NAMECONSTRAINT_STR
EOF

# --------------------------------------------
# 4. Generate the Root CA certificate
# --------------------------------------------
echo "Generating restricted Root CA certificate..."
openssl req -x509 -new -nodes -key ca.key -sha256 -days $VALIDITY_DAYS \
    -out ca.crt -config ca.cnf

rm ca.cnf

# --------------------------------------------
# 5. Generate Server Certificate private key
# --------------------------------------------
echo "Generating server key..."
openssl genrsa -out server.key 2048

# --------------------------------------------
# 6. Create config for server CSR
# --------------------------------------------
cat >server.cnf <<EOF
[ req ]
default_bits       = 2048
distinguished_name = req_distinguished_name
req_extensions     = v3_req
prompt             = no

[ req_distinguished_name ]
CN = ${DOMAINS[0]}

[ v3_req ]
subjectAltName = $SAN_STR
EOF

# --------------------------------------------
# 7. Generate CSR
# --------------------------------------------
echo "Generating server certificate signing request (CSR)..."
openssl req -new -key server.key -out server.csr -config server.cnf

# --------------------------------------------
# 8. Create config for CA signing
# --------------------------------------------
cat >sign.cnf <<EOF
[ ca ]
default_ca = CA_default

[ CA_default ]
database    = index.txt
serial      = serial
new_certs_dir = .
default_md  = sha256
policy      = policy_any

[ policy_any ]
commonName              = supplied
organizationName        = optional
organizationalUnitName  = optional
countryName             = optional
stateOrProvinceName     = optional
localityName            = optional
emailAddress            = optional

[ v3_server ]
subjectAltName = $SAN_STR
EOF

# --------------------------------------------
# 9. Create CA database files
# --------------------------------------------
touch index.txt
echo 01 >serial

# --------------------------------------------
# 10. Sign the server certificate
# --------------------------------------------
echo "Signing server certificate with the restricted CA..."
openssl ca -batch -in server.csr -out server.crt -keyfile ca.key -cert ca.crt \
    -config sign.cnf -extensions v3_server -days $VALIDITY_DAYS -notext

# --------------------------------------------
# 11. Clean up temporary files
# --------------------------------------------
rm -f server.csr index.txt index.txt.attr serial serial.old index.txt.old \
    server.cnf sign.cnf 01.pem *.pem 2>/dev/null || true

# --------------------------------------------
# 12. Optionally delete the Root CA private key
#     (Uncomment the next line to nuke it)
# --------------------------------------------
# rm -f ca.key

echo ""
echo "------------------------------------------------------------"
echo "SUCCESS!"
echo "------------------------------------------------------------"
echo "Root CA (install on phone/other devices): ca.crt"
echo "Server certificate (use in Caddy):         server.crt"
echo "Server private key (use in Caddy):         server.key"
if [ -f ca.key ]; then
    echo "Root CA private key (keep safe):            ca.key"
    echo ""
    echo "If you never need to issue another certificate, you can"
    echo "safely delete ca.key to prevent any future misuse:"
    echo "  rm ca.key"
else
    echo "Root CA private key has been deleted (max security)."
fi
echo ""
echo "This Root CA can ONLY issue valid certificates for:"
for entry in "${DOMAINS[@]}"; do
    if [[ "$entry" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || [[ "$entry" =~ : ]]; then
        echo "  - IP: $entry (exact)"
    else
        echo "  - $entry"
        echo "  - *.${entry}"
    fi
done
echo ""
echo "The server certificate is valid for:"
echo "  - ${DOMAINS[*]}"
echo "  - localhost, 127.0.0.1, ::1"
echo "------------------------------------------------------------"
