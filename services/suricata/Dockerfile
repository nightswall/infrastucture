# this is a docker file that will build a container that runs suricata IDS and will have a volume mounted to /data that will contain the rules and logs
# and will also have a volume mounted to /pcap that will contain the pcap files to be analyzed by suricata
# this container will also run telegraf and will send the logs to influxdb and will send the metrics to influxdb as well

FROM debian:stretch


RUN apt-get update && apt-get install -y \
    suricata \
    libpcre3 \
    libpcre3-dbg \
    libpcre3-dev \
    build-essential \
    libpcap-dev \
    libnet1-dev \
    libyaml-0-2 \
    libyaml-dev \
    zlib1g \
    zlib1g-dev \
    libcap-ng-dev \
    libcap-ng0 \
    make \
    libmagic-dev \
    libjansson-dev \
    libjansson4 \
    pkg-config \
    wget \
    python-pip \
    python-setuptools \
    python-dev \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libgeoip-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install --upgrade setuptools \
    && pip install --upgrade pyinotify

# set up suricata user
RUN groupadd -r suricata && useradd -r -g suricata suricata

# install suricata-update
RUN pip install --upgrade suricata-update

# install suricata rules
RUN suricata-update

# install telegraf
RUN curl -O https://dl.influxdata.com/telegraf/releases/telegraf_1.21.4-1_amd64.deb \
    && dpkg -i telegraf_1.21.4-1_amd64.deb

# setup suricata
RUN mkdir -p /data /pcap
COPY suricata.yaml /etc/suricata/suricata.yaml
COPY suricata /etc/default/suricata

# setup telegraf
COPY telegraf.conf /etc/telegraf/telegraf.conf

RUN mkdir -p /data/pcap
COPY start.sh /start.sh
RUN chmod +x /start.sh

VOLUME ["/data", "/pcap"]
#RUN service suricata start
WORKDIR /
CMD ["./start.sh"]
#CMD service suricata start && /bin/bash
#CMD ["telegraf","--config", "/etc/telegraf/telegraf.conf"]