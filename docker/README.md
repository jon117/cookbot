# Isaac Sim in Container

https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html

When you run their nvcr.io/nvidia/isaac-sim:5.0.0 container, this happens:

```
Failed to open [/var/run/utmp]
Active user not found. Using default user [kiosk]
[Error] [carb.livestream-rtc.plugin] Stream Server: Net Stream Creation failed, 0x800E8401
[Error] [carb.livestream-rtc.plugin] Could not initialize streaming components
[Error] [carb.livestream-rtc.plugin] Couldn't initialize the capture device.
```

it needs the package libh264-dev to enable the webRTC streaming. so the Dockerfile pulls this image, installs the h264 library.

## Docker Compose convenience 

run the modified (correct) container using:

```
docker compose up --build
```

## WebRTC client

Download:
https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#isaac-sim-latest-release


On ubuntu24.04:

```
./isaacsim-webrtc-streaming-client-1.1.4-linux-x64.AppImage --appimage-extract
cd squashfs-root
sudo chown root chrome-sandbox
sudo chmod 4755 chrome-sandbox
./isaac-sim-webrtc-streaming-client
```

Enter the IP of the container (or localhost), voila!

## To Do:

How to yeet python into this container? so you can "deploy" to it?
That'd be dope.
