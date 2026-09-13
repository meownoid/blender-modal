# syntax=docker/dockerfile:1.7

FROM ubuntu:24.04 AS alembic-builder
ARG ALEMBIC_REF=1.8.5
ARG ALEMBIC_COMMIT=f4e3cc33e4270ade9d076a2cac7a2a4409c61675
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential ca-certificates cmake git libimath-dev
RUN rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "$ALEMBIC_REF" https://github.com/alembic/alembic.git /build/alembic
RUN test "$(git -C /build/alembic rev-parse HEAD)" = "$ALEMBIC_COMMIT"
RUN cmake -S /build/alembic -B /build/alembic-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/alembic -DUSE_BINARIES=OFF -DUSE_TESTS=OFF
RUN cmake --build /build/alembic-build --parallel 2
RUN cmake --install /build/alembic-build

FROM ubuntu:24.04 AS flip-fluids-builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential cmake libimath-dev
RUN rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY --from=alembic-builder /opt/alembic /opt/alembic
COPY third_party/flip-fluids/ ./
RUN cmake -S . -B build -DALEMBIC_PACKAGE_ROOT=/opt/alembic -DDISTRIBUTE_SOURCE=OFF
RUN cmake --build build --parallel 2

FROM ubuntu:24.04 AS runtime
ARG BLENDER_VERSION=5.2.1
ARG BLENDER_SHA256=a31f524fa99a527d3d52b7f5aaa68c34e1a19d5a1c9473f79c5cc610fd5b10e9
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV FLIP_FLUIDS_ADDON=flip_fluids_addon
RUN apt-get update
RUN apt-get install -y --no-install-recommends ca-certificates curl xz-utils python3 python3-pip python3-venv build-essential libgomp1 libimath-3-1-29t64 libegl1 libgl1 libsm6 libx11-6 libxext6 libxfixes3 libxi6 libxkbcommon0 libxrender1 libxxf86vm1 libwayland-client0
RUN curl -fsSLo /tmp/blender.tar.xz "https://download.blender.org/release/Blender5.2/blender-$BLENDER_VERSION-linux-x64.tar.xz"
RUN echo "$BLENDER_SHA256  /tmp/blender.tar.xz" | sha256sum -c -
RUN mkdir -p /opt/blender
RUN tar -xJf /tmp/blender.tar.xz --strip-components=1 -C /opt/blender
RUN rm /tmp/blender.tar.xz
RUN find /opt/blender -type f -name '*.a' -delete
RUN rm -rf /opt/blender/5.2/datafiles/locale /var/lib/apt/lists/*
RUN python3 -m venv /app/.venv
RUN /app/.venv/bin/pip install --no-cache-dir --upgrade pip
ENV PATH=/app/.venv/bin:$PATH
COPY --from=flip-fluids-builder /build/build/bl_flip_fluids/flip_fluids_addon/ /opt/blender/5.2/scripts/addons_core/flip_fluids_addon/
COPY --from=alembic-builder /opt/alembic/lib/ /opt/blender/5.2/scripts/addons_core/flip_fluids_addon/ffengine/lib/
COPY renderer/blendrender_enable_flip_fluids.py /tmp/blendrender_enable_flip_fluids.py
RUN /opt/blender/blender --background --factory-startup --disable-autoexec --python-exit-code 1 --python /tmp/blendrender_enable_flip_fluids.py --python-expr "from flip_fluids_addon.ffengine.ffengine import ffengine; assert ffengine.FluidSimulation_get_version"

WORKDIR /app
COPY pyproject.toml README.md Dockerfile .dockerignore ./
COPY blender_modal/ ./blender_modal/
COPY renderer/ ./renderer/
COPY third_party/flip-fluids/ ./third_party/flip-fluids/
RUN pip install --no-cache-dir .
RUN groupadd --gid 10001 blender
RUN useradd --uid 10001 --gid blender --no-create-home --shell /usr/sbin/nologin blender
RUN mkdir -p /opt/blender/5.2/scripts/addons_core/flip_fluids_addon/materials/material_library/icons
RUN chown -R blender:blender /app /opt/blender/5.2/scripts/addons_core/flip_fluids_addon/materials/material_library/icons /opt/blender/5.2/scripts/addons_core/flip_fluids_addon/resources/installation_data
# Modal mounts Volumes writable by the function runtime user. Do not switch users
# here: doing so would make the durable renderer catalog read-only.
