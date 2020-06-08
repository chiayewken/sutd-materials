#!/usr/bin/env bash
whoami
mkdir -p data
cd data
wget -nc http://mattmahoney.net/dc/text8.zip
unzip -qn text8.zip
