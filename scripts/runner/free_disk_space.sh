#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#
# The Azure provided machines typically have the following disk allocation:
# Total space: 85GB
# Allocated: 67 GB
# Free: 17 GB
# This script frees up 28 GB of disk space by deleting unneeded packages and 
# large directories.
# The Flink end to end tests download and generate more than 17 GB of files,
# causing unpredictable behavior and build failures.
#
echo "=============================================================================="
echo "Freeing up disk space on CI system"
echo "=============================================================================="

echo "Listing 100 largest packages"
dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -n | tail -n 100
df -h
echo "Removing large packages"
sudo apt-get remove -y '^ghc-8.*'
sudo apt-get remove -y '^dotnet-.*'
sudo apt-get remove -y '^llvm-.*'
sudo apt-get remove -y 'php.*'
sudo apt-get remove -y azure-cli google-cloud-sdk hhvm google-chrome-stable firefox powershell mono-devel
sudo apt-get autoremove -y
sudo apt-get clean
df -h
echo "Removing large directories"
# deleting 15GB
rm -rf /usr/share/dotnet/

# Another source :
df -h
echo "Now moving to another source"

sudo docker rmi $(docker image ls -aq) >/dev/null 2>&1 || true
sudo rm -rf \
  /usr/share/dotnet /usr/local/lib/android /opt/ghc \
  /usr/local/share/powershell /usr/share/swift /usr/local/.ghcup \
  /usr/lib/jvm || true
echo "some directories deleted"
sudo apt install aptitude -y >/dev/null 2>&1
sudo aptitude purge aria2 ansible azure-cli shellcheck rpm xorriso zsync \
  esl-erlang firefox gfortran-8 gfortran-9 google-chrome-stable \
  imagemagick \
  libmagickcore-dev libmagickwand-dev libmagic-dev ant ant-optional kubectl \
  mercurial apt-transport-https mono-complete libmysqlclient \
  unixodbc-dev yarn chrpath libssl-dev libxft-dev \
  libfreetype6 libfreetype6-dev libfontconfig1 libfontconfig1-dev \
  snmp pollinate libpq-dev postgresql-client powershell ruby-full \
  sphinxsearch subversion mongodb-org azure-cli microsoft-edge-stable \
  -y -f >/dev/null 2>&1
sudo aptitude purge microsoft-edge-stable -f -y >/dev/null 2>&1 || true
sudo apt purge microsoft-edge-stable -f -y >/dev/null 2>&1 || true
sudo aptitude purge '~n ^mysql' -f -y >/dev/null 2>&1
sudo aptitude purge '~n ^php' -f -y >/dev/null 2>&1
sudo aptitude purge '~n ^dotnet' -f -y >/dev/null 2>&1
sudo apt-get autoremove -y >/dev/null 2>&1
sudo apt-get autoclean -y >/dev/null 2>&1
echo "some packages purged"
df -h