FROM python:3.12

# The shell tool needs tmux. Installing it at runtime races with apt's lock
# and depends on the container reaching Debian's mirrors.
RUN sed -i \
      -e 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
      -e 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
      /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tmux \
    && rm -rf /var/lib/apt/lists/*
