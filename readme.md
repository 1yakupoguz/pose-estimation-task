
# Adastec HVAC Manager

This repository contains a ROS Noetic-based project simulating a vehicle HVAC system using a virtual CAN network. The core of this system is the `hvac_manager` node, which subscribes to incoming CAN messages, processes HVAC-related signals, and publishes corresponding commands.

---

## 📌 Project Overview

This project was developed as part of the Adastec ROS Development Assignment. It includes:

- A ROS node written in C++ that manages HVAC logic
- Parameterized bit configurations via YAML
- CAN message handling using SocketCAN bridge
- Automation with `install.sh` script
- Virtual CAN network setup and log replay support

---

## ⚙️ Setup Instructions

### 1. Clone and Install

```bash
git clone https://github.com/1yakupoguz/adastec_ws.git
cd adastec_ws
chmod +x install.sh
./install.sh
```

This script installs dependencies, sets up ROS environment, builds the workspace, and initializes a virtual CAN interface.

---

## 🚀 How to Launch

```bash
canplayer -I src/data/adastec_can.log

roslaunch hvac_manager default.launch
```

This launches:

- `socketcan_to_topic_node`
- `topic_to_socketcan_node`
- `hvac_manager_node`

---

## 📡 Node Description

### 📥 Subscriptions

- `/received_messages` — Incoming CAN frames
- `/hvac_commands` — Optional override commands (std_msgs/UInt8MultiArray)

### 📤 Publications

- `/sent_messages` — Constructed HVAC command messages (can_msgs/Frame)

---

## 🧠 Node Logic

- Extracts HVAC input signals from CAN ID `1568`
- If `heating_input` is active:
  - Sets fan mode to `Heating` (value `1`)
- If `cooling_input` is active:
  - Sets fan mode to `Cooling` (value `2`)
- Sets fan speed and calorifer speed commands accordingly
- Publishes result as a CAN frame with ID `1559`

---

## ⚙️ Parameterization

Located in `params/params.yaml`:

```yaml
hvac_manager_node:
  message_ids:
    input_status_msg_id: 1568
    autonomous_hvac_cmd_msg_id: 1559
  signal_bits:
    passenger_calorifer_speed_input:
      byte_index: 3
      bit_offset: 0
      length: 2
    ...
```

All bit-level configurations for extracting and encoding signals are read dynamically at runtime.

---

## 🛠️ Development Notes

- Developed using ROS Noetic (Ubuntu 20.04)
- Built with `catkin_tools` (`catkin build`)
- Uses SocketCAN bridge (`socketcan_bridge`) for CAN communication
- Fully modular with separation of launch, param, and source files

---

## 📜 Example Override Command

```bash
rostopic pub /hvac_commands std_msgs/UInt8MultiArray "data: [1, 2, 1]"
```

This sets mode to `Heating`, driver fan speed to level `2`, and passenger calorifer speed to level `1`.

---
