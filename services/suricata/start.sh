#!/bin/bash
service suricata start
sleep 5
telegraf --config /etc/telegraf/telegraf.conf