#!/usr/bin/env bash

# USB udev 规则创建助手。
#
# 仅处理能映射为以下设备节点的 USB 设备：
#   /dev/ttyUSB* /dev/ttyACM* /dev/video* /dev/input/js* /dev/hidraw*
# 用户必须先选择一个具体节点，再为它创建稳定别名。

set -Eeuo pipefail

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'
readonly TEMP_DIR="$(mktemp -d /tmp/udev-helper.XXXXXX)"
readonly TEMP_RULE_FILE="$TEMP_DIR/rule.rules"

declare -a CANDIDATE_NODES=()
declare -a CANDIDATE_SUBSYSTEMS=()
declare -a CANDIDATE_KERNEL_PATTERNS=()
declare -a CANDIDATE_ALIAS_DIRS=()
declare -a CANDIDATE_USB_PATHS=()
declare -a CANDIDATE_VIDS=()
declare -a CANDIDATE_PIDS=()
declare -a CANDIDATE_SERIALS=()
declare -a CANDIDATE_INTERFACES=()
declare -a CANDIDATE_VIDEO_INDEXES=()
declare -a CANDIDATE_DESCRIPTIONS=()
declare -a SYSTEM_RULE_FILES=()
EXCLUDED_USB_KERNEL=""

cleanup() {
    rm -rf -- "$TEMP_DIR"
}
trap cleanup EXIT

print_info() { printf '%b[INFO]%b %s\n' "$BLUE" "$NC" "$*"; }
print_success() { printf '%b[SUCCESS]%b %s\n' "$GREEN" "$NC" "$*"; }
print_warning() { printf '%b[WARNING]%b %s\n' "$YELLOW" "$NC" "$*"; }
print_error() { printf '%b[ERROR]%b %s\n' "$RED" "$NC" "$*" >&2; }

die() {
    print_error "$*"
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "缺少命令：$1"
}

prompt() {
    local message="$1"
    local value
    if ! IFS= read -r -p "$message" value; then
        die "未读取到输入"
    fi
    printf '%s' "$value"
}

# 返回指定 sysfs 目录中的单行属性。属性不存在时返回非零状态。
sys_attr() {
    local sys_path="$1"
    local attribute="$2"
    local attribute_path="$sys_path/$attribute"

    [[ -r "$attribute_path" ]] || return 1
    tr -d '\n' < "$attribute_path"
}

# 返回 udev 属性值，例如 get_property /dev/ttyUSB0 ID_USB_INTERFACE_NUM。
get_property() {
    local node="$1"
    local requested_key="$2"
    local line key value

    while IFS= read -r line; do
        key="${line%%=*}"
        if [[ "$key" == "$requested_key" ]]; then
            value="${line#*=}"
            printf '%s' "$value"
            return 0
        fi
    done < <(udevadm info --query=property --name="$node" 2>/dev/null)

    return 1
}

# 查找设备节点对应的 USB 设备父目录。返回的路径形如
# /sys/devices/.../usb1/1-2.3，而非 /dev 节点或不适用于 KERNELS 的 DEVPATH。
find_usb_parent() {
    local node="$1"
    local devpath current

    devpath="$(udevadm info --query=path --name="$node" 2>/dev/null)" || return 1
    current="/sys$devpath"

    while [[ "$current" != "/sys" && "$current" != "/" ]]; do
        if [[ -r "$current/idVendor" && -r "$current/idProduct" ]]; then
            printf '%s' "$current"
            return 0
        fi
        current="$(dirname "$current")"
    done

    return 1
}

# 为一个已知支持的节点确定规则中的目标子系统、稳定内核通配符和别名目录。
describe_node_type() {
    local node="$1"

    case "$node" in
        /dev/ttyUSB*)
            NODE_SUBSYSTEM="tty"
            NODE_KERNEL_PATTERN="ttyUSB*"
            NODE_ALIAS_DIR=""
            ;;
        /dev/ttyACM*)
            NODE_SUBSYSTEM="tty"
            NODE_KERNEL_PATTERN="ttyACM*"
            NODE_ALIAS_DIR=""
            ;;
        /dev/video*)
            NODE_SUBSYSTEM="video4linux"
            NODE_KERNEL_PATTERN="video*"
            NODE_ALIAS_DIR=""
            ;;
        /dev/input/js*)
            NODE_SUBSYSTEM="input"
            NODE_KERNEL_PATTERN="js*"
            NODE_ALIAS_DIR="input"
            ;;
        /dev/hidraw*)
            NODE_SUBSYSTEM="hidraw"
            NODE_KERNEL_PATTERN="hidraw*"
            NODE_ALIAS_DIR=""
            ;;
        *)
            return 1
            ;;
    esac
}

append_candidate() {
    local node="$1"
    local usb_path vid pid serial interface video_index manufacturer product description

    describe_node_type "$node" || return 0
    usb_path="$(find_usb_parent "$node")" || return 0
    vid="$(sys_attr "$usb_path" idVendor)" || return 0
    pid="$(sys_attr "$usb_path" idProduct)" || return 0
    serial="$(sys_attr "$usb_path" serial 2>/dev/null || true)"
    interface="$(get_property "$node" ID_USB_INTERFACE_NUM 2>/dev/null || true)"
    video_index=""
    if [[ "$NODE_SUBSYSTEM" == "video4linux" ]]; then
        local node_path
        node_path="/sys$(udevadm info --query=path --name="$node" 2>/dev/null)"
        video_index="$(sys_attr "$node_path" index 2>/dev/null || true)"
    fi

    manufacturer="$(sys_attr "$usb_path" manufacturer 2>/dev/null || true)"
    product="$(sys_attr "$usb_path" product 2>/dev/null || true)"
    description="${manufacturer:+$manufacturer }${product:-USB device}"

    CANDIDATE_NODES+=("$node")
    CANDIDATE_SUBSYSTEMS+=("$NODE_SUBSYSTEM")
    CANDIDATE_KERNEL_PATTERNS+=("$NODE_KERNEL_PATTERN")
    CANDIDATE_ALIAS_DIRS+=("$NODE_ALIAS_DIR")
    CANDIDATE_USB_PATHS+=("$usb_path")
    CANDIDATE_VIDS+=("${vid,,}")
    CANDIDATE_PIDS+=("${pid,,}")
    CANDIDATE_SERIALS+=("$serial")
    CANDIDATE_INTERFACES+=("$interface")
    CANDIDATE_VIDEO_INDEXES+=("$video_index")
    CANDIDATE_DESCRIPTIONS+=("$description")
}

scan_usb_nodes() {
    local node

    print_info "正在扫描 USB 设备节点…"
    shopt -s nullglob
    for node in /dev/ttyUSB* /dev/ttyACM* /dev/video* /dev/input/js* /dev/hidraw*; do
        [[ -e "$node" || -L "$node" ]] || continue
        append_candidate "$node"
    done
    shopt -u nullglob

    ((${#CANDIDATE_NODES[@]} > 0)) || die "未找到受支持的 USB 设备节点"
}

select_candidate() {
    local index choice serial interface

    printf '\n%-4s %-22s %-11s %-16s %s\n' '序号' '设备节点' 'VID:PID' '序列号' '描述'
    printf '%s\n' '---- ---------------------- ----------- ---------------- ------------------------------'
    for index in "${!CANDIDATE_NODES[@]}"; do
        serial="${CANDIDATE_SERIALS[index]:--}"
        interface="${CANDIDATE_INTERFACES[index]:-}"
        if [[ -n "$interface" ]]; then
            serial="${serial} (IF ${interface})"
        fi
        printf '%-4d %-22s %-11s %-16.16s %s\n' \
            "$((index + 1))" \
            "${CANDIDATE_NODES[index]}" \
            "${CANDIDATE_VIDS[index]}:${CANDIDATE_PIDS[index]}" \
            "$serial" \
            "${CANDIDATE_DESCRIPTIONS[index]}"
    done

    while true; do
        choice="$(prompt '请选择要创建别名的设备节点序号： ')"
        if [[ "$choice" =~ ^[0-9]+$ ]] && ((choice >= 1 && choice <= ${#CANDIDATE_NODES[@]})); then
            SELECTED_INDEX=$((choice - 1))
            print_info "选中节点的物理 USB 端口：$(basename "${CANDIDATE_USB_PATHS[SELECTED_INDEX]}")"
            return
        fi
        print_warning "请输入表中的有效序号"
    done
}

select_match_strategy() {
    local choice default_choice
    local serial="${CANDIDATE_SERIALS[SELECTED_INDEX]}"

    if [[ -n "$serial" ]]; then
        default_choice=1
    else
        default_choice=2
    fi

    printf '\n匹配方式：\n'
    if [[ -n "$serial" ]]; then
        printf '  1) VID:PID + 序列号（推荐；设备可换 USB 口）\n'
    else
        printf '  1) VID:PID + 序列号（此设备没有序列号，不能使用）\n'
    fi
    printf '  2) VID:PID + 物理 USB 端口（设备须插在同一端口）\n'
    printf '  3) 仅 VID:PID（高级选项：同型号设备会争用同一别名）\n'
    printf '  4) VID:PID + 排除一个物理 USB 端口（高级选项）\n'

    while true; do
        choice="$(prompt "请选择匹配方式 [默认 $default_choice]： ")"
        choice="${choice:-$default_choice}"
        case "$choice" in
            1)
                if [[ -n "$serial" ]]; then
                    MATCH_STRATEGY="serial"
                    return
                fi
                print_warning "选中的设备没有可用序列号，请选择方式 2 或 3"
                ;;
            2)
                MATCH_STRATEGY="physical"
                return
                ;;
            3)
                print_warning "仅 VID:PID 会匹配所有同型号的受支持节点，别名最终指向哪一个并不可靠。"
                if [[ "$(prompt '仍要继续吗？(y/N): ')" =~ ^[Yy]$ ]]; then
                    MATCH_STRATEGY="vidpid"
                    return
                fi
                ;;
            4)
                print_warning "此方式会匹配除一个固定端口外的所有同型号设备；第三台同型号设备仍可能争用别名。"
                select_excluded_usb_port
                MATCH_STRATEGY="vidpid_excluding_port"
                return
                ;;
            *)
                print_warning "请输入 1、2、3 或 4"
                ;;
        esac
    done
}

select_excluded_usb_port() {
    local index candidate_kernel selected_kernel choice manual_kernel seen
    local -a available_kernels=()
    local selected_vid="${CANDIDATE_VIDS[SELECTED_INDEX]}"
    local selected_pid="${CANDIDATE_PIDS[SELECTED_INDEX]}"

    selected_kernel="$(basename "${CANDIDATE_USB_PATHS[SELECTED_INDEX]}")"
    printf '\n可排除的同型号物理 USB 端口：\n'
    printf '  1) 手动输入端口名（例如 1-2.3）\n'

    for index in "${!CANDIDATE_NODES[@]}"; do
        [[ "${CANDIDATE_VIDS[index]}" == "$selected_vid" ]] || continue
        [[ "${CANDIDATE_PIDS[index]}" == "$selected_pid" ]] || continue
        candidate_kernel="$(basename "${CANDIDATE_USB_PATHS[index]}")"
        [[ "$candidate_kernel" == "$selected_kernel" ]] && continue

        seen=false
        for manual_kernel in "${available_kernels[@]}"; do
            if [[ "$manual_kernel" == "$candidate_kernel" ]]; then
                seen=true
                break
            fi
        done
        if [[ "$seen" == false ]]; then
            available_kernels+=("$candidate_kernel")
            printf '  %d) %s\n' "$((${#available_kernels[@]} + 1))" "$candidate_kernel"
        fi
    done

    while true; do
        choice="$(prompt '请选择要排除的端口： ')"
        if [[ "$choice" == "1" ]]; then
            manual_kernel="$(prompt '请输入要排除的 USB 端口名（例如 1-2.3）： ')"
            if [[ "$manual_kernel" =~ ^[0-9]+-[0-9]+([.][0-9]+)*$ ]]; then
                if [[ "$manual_kernel" == "$selected_kernel" ]]; then
                    print_warning "不能排除当前所选节点所在端口，否则规则永远不会匹配"
                else
                    EXCLUDED_USB_KERNEL="$manual_kernel"
                    return
                fi
            else
                print_warning "端口名格式无效，应类似 1-2 或 1-2.3"
            fi
        elif [[ "$choice" =~ ^[0-9]+$ ]] && ((choice >= 2 && choice < ${#available_kernels[@]} + 2)); then
            EXCLUDED_USB_KERNEL="${available_kernels[choice - 2]}"
            return
        else
            print_warning "请输入列表中的有效序号"
        fi
    done
}

select_alias() {
    local alias alias_dir="${CANDIDATE_ALIAS_DIRS[SELECTED_INDEX]}"

    while true; do
        alias="$(prompt '请输入别名（字母、数字、_、-）： ')"
        if [[ "$alias" =~ ^[[:alnum:]_-]+$ ]]; then
            if [[ -n "$alias_dir" ]]; then
                RELATIVE_ALIAS="$alias_dir/$alias"
            else
                RELATIVE_ALIAS="$alias"
            fi
            ALIAS_PATH="/dev/$RELATIVE_ALIAS"
            RULE_STEM="${RELATIVE_ALIAS//\//_}"
            return
        fi
        print_warning "别名不能为空，且只能使用字母、数字、_、-"
    done
}

udev_quote() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/}"
    printf '%s' "$value"
}

generate_rule() {
    local index="$SELECTED_INDEX"
    local node="${CANDIDATE_NODES[index]}"
    local subsystem="${CANDIDATE_SUBSYSTEMS[index]}"
    local kernel_pattern="${CANDIDATE_KERNEL_PATTERNS[index]}"
    local usb_path="${CANDIDATE_USB_PATHS[index]}"
    local vid="${CANDIDATE_VIDS[index]}"
    local pid="${CANDIDATE_PIDS[index]}"
    local serial="${CANDIDATE_SERIALS[index]}"
    local interface="${CANDIDATE_INTERFACES[index]}"
    local video_index="${CANDIDATE_VIDEO_INDEXES[index]}"
    local usb_kernel
    local -a conditions=()

    usb_kernel="$(basename "$usb_path")"
    conditions+=("SUBSYSTEM==\"$(udev_quote "$subsystem")\"")
    conditions+=("KERNEL==\"$(udev_quote "$kernel_pattern")\"")
    conditions+=("ATTRS{idVendor}==\"$(udev_quote "$vid")\"")
    conditions+=("ATTRS{idProduct}==\"$(udev_quote "$pid")\"")

    case "$MATCH_STRATEGY" in
        serial)
            conditions+=("ATTRS{serial}==\"$(udev_quote "$serial")\"")
            MATCH_DESCRIPTION="VID:PID + 序列号"
            ;;
        physical)
            # KERNELS 只接受内核设备名（例如 1-2.3），不能传入 DEVPATH。
            conditions+=("KERNELS==\"$(udev_quote "$usb_kernel")\"")
            MATCH_DESCRIPTION="VID:PID + 物理 USB 端口 ($usb_kernel)"
            ;;
        vidpid)
            MATCH_DESCRIPTION="仅 VID:PID（高级选项）"
            ;;
        vidpid_excluding_port)
            conditions+=("KERNELS!=\"$(udev_quote "$EXCLUDED_USB_KERNEL")\"")
            MATCH_DESCRIPTION="VID:PID + 排除物理 USB 端口 ($EXCLUDED_USB_KERNEL)"
            ;;
        *)
            die "未知匹配方式：$MATCH_STRATEGY"
            ;;
    esac

    # 接口编号可防止一台复合 USB 设备的不同接口共享同一别名。它由系统的
    # usb_id udev 规则在本脚本生成的 99 号规则之前设置。
    if [[ "$MATCH_STRATEGY" != "vidpid" && "$MATCH_STRATEGY" != "vidpid_excluding_port" && -n "$interface" ]]; then
        conditions+=("ENV{ID_USB_INTERFACE_NUM}==\"$(udev_quote "$interface")\"")
    fi

    # 单个摄像头接口可能暴露多个 video 节点；index 是其稳定功能编号。
    if [[ "$MATCH_STRATEGY" != "vidpid" && "$MATCH_STRATEGY" != "vidpid_excluding_port" && -n "$video_index" ]]; then
        conditions+=("ATTR{index}==\"$(udev_quote "$video_index")\"")
    fi

    {
        printf '# USB udev rule generated by udev.sh\n'
        printf '# Selected node: %s\n' "$node"
        printf '# Alias: %s\n' "$ALIAS_PATH"
        printf '# Match strategy: %s\n' "$MATCH_DESCRIPTION"
        printf '# Generated: %s\n\n' "$(date --iso-8601=seconds)"
        printf 'ACTION=="add"'
        printf ', %s' "${conditions[@]}"
        printf ', SYMLINK+="%s", MODE:="0666"\n' "$(udev_quote "$RELATIVE_ALIAS")"
    } > "$TEMP_RULE_FILE"

    printf '\n生成的规则：\n%s\n' '========================================'
    cat "$TEMP_RULE_FILE"
    printf '%s\n' '========================================'
}

collect_system_rule_files() {
    local file

    SYSTEM_RULE_FILES=()
    [[ -d /etc/udev/rules.d ]] || return

    shopt -s nullglob
    for file in /etc/udev/rules.d/*.rules; do
        [[ -f "$file" ]] && SYSTEM_RULE_FILES+=("$file")
    done
    shopt -u nullglob
}

select_destination_rule_file() {
    local choice index file_name default_file

    collect_system_rule_files
    default_file="99-udev-helper-${RULE_STEM}.rules"

    printf '\n规则文件目标：\n'
    printf '  1) 新建规则文件（默认：/etc/udev/rules.d/%s）\n' "$default_file"
    for index in "${!SYSTEM_RULE_FILES[@]}"; do
        printf '  %d) 追加到 %s\n' "$((index + 2))" "${SYSTEM_RULE_FILES[index]}"
    done

    while true; do
        choice="$(prompt '请选择目标 [默认 1]： ')"
        choice="${choice:-1}"
        if [[ "$choice" =~ ^[0-9]+$ ]] && ((choice == 1)); then
            file_name="$(prompt "请输入新文件名 [默认 $default_file]： ")"
            file_name="${file_name:-$default_file}"
            if [[ ! "$file_name" =~ ^[[:alnum:]_][[:alnum:]_.-]*\.rules$ ]]; then
                print_warning "文件名必须以 .rules 结尾，且只能使用字母、数字、_、.、-"
                continue
            fi
            DESTINATION_RULE_FILE="/etc/udev/rules.d/$file_name"
            if [[ -e "$DESTINATION_RULE_FILE" || -L "$DESTINATION_RULE_FILE" ]]; then
                print_warning "文件已存在，请在列表中选择“追加到”该文件"
                continue
            fi
            DESTINATION_MODE="create"
            return
        fi

        if [[ "$choice" =~ ^[0-9]+$ ]] && ((choice >= 2 && choice < ${#SYSTEM_RULE_FILES[@]} + 2)); then
            DESTINATION_RULE_FILE="${SYSTEM_RULE_FILES[choice - 2]}"
            DESTINATION_MODE="append"
            return
        fi

        print_warning "请输入列表中的有效序号"
    done
}

run_privileged() {
    if ((EUID == 0)); then
        "$@"
    else
        command -v sudo >/dev/null 2>&1 || die "应用规则需要 root 权限，但系统未安装 sudo"
        sudo "$@"
    fi
}

refresh_udev_rules() {
    printf '\n%s\n' '将重新加载 udev 规则，并对当前设备重放 add 事件。'
    print_warning '这不会写入任何规则文件，但可能执行现有规则中的 RUN+= 操作。'

    if [[ ! "$(prompt '是否继续刷新？(y/N): ')" =~ ^[Yy]$ ]]; then
        print_info "未刷新规则"
        return
    fi

    run_privileged udevadm control --reload-rules
    run_privileged udevadm trigger --action=add
    run_privileged udevadm settle --timeout=5
    print_success "udev 规则已重新加载，当前设备已重新触发"
}

apply_rule() {
    local selected_node="${CANDIDATE_NODES[SELECTED_INDEX]}"
    local target_path

    select_destination_rule_file

    if [[ -e "$ALIAS_PATH" || -L "$ALIAS_PATH" ]]; then
        target_path="$(readlink -f -- "$ALIAS_PATH" 2>/dev/null || true)"
        print_warning "别名路径已存在：$ALIAS_PATH${target_path:+ -> $target_path}"
    fi
    if [[ "$DESTINATION_MODE" == "append" ]]; then
        print_info "将把规则追加到：$DESTINATION_RULE_FILE"
    else
        print_info "将新建规则文件：$DESTINATION_RULE_FILE"
    fi

    if [[ ! "$(prompt '是否立即安装并启用此规则？(y/N): ')" =~ ^[Yy]$ ]]; then
        print_info "规则未安装。临时规则文件将在脚本退出时清理。"
        return
    fi

    if [[ "$DESTINATION_MODE" == "append" ]]; then
        # 先写入一个空行，避免目标文件末尾没有换行时把两条规则连在一起。
        printf '\n' | run_privileged tee -a "$DESTINATION_RULE_FILE" >/dev/null
        run_privileged tee -a "$DESTINATION_RULE_FILE" < "$TEMP_RULE_FILE" >/dev/null
    else
        run_privileged install -D -m 0644 "$TEMP_RULE_FILE" "$DESTINATION_RULE_FILE"
    fi
    run_privileged udevadm control --reload-rules
    # 仅重放当前所选节点的 add 事件，避免当前脚本的规则安装影响其他设备。
    run_privileged udevadm trigger --action=add \
        --subsystem-match="${CANDIDATE_SUBSYSTEMS[SELECTED_INDEX]}" \
        --sysname-match="$(basename "$selected_node")"
    run_privileged udevadm settle --timeout=5

    if [[ "$DESTINATION_MODE" == "append" ]]; then
        print_success "规则已追加到：$DESTINATION_RULE_FILE"
    else
        print_success "规则已创建：$DESTINATION_RULE_FILE"
    fi
    if [[ -L "$ALIAS_PATH" ]]; then
        print_success "别名已创建：$ALIAS_PATH -> $(readlink -f -- "$ALIAS_PATH")"
    else
        print_warning "规则已安装，但 $ALIAS_PATH 尚未出现；请重新插拔设备后检查。"
    fi
}

main() {
    local action

    require_command udevadm
    require_command install
    require_command tee
    require_command dirname
    require_command basename

    printf '\n%s\n' '=========================================='
    printf '%s\n' '       USB udev 规则创建助手'
    printf '%s\n\n' '=========================================='

    while true; do
        printf '1) 创建并安装新的 udev 规则\n'
        printf '2) 仅刷新已安装的 udev 规则\n'
        action="$(prompt '请选择操作 [默认 1]： ')"
        action="${action:-1}"
        case "$action" in
            1)
                scan_usb_nodes
                select_candidate
                select_match_strategy
                select_alias
                print_info "将为 ${CANDIDATE_NODES[SELECTED_INDEX]} 创建别名 $ALIAS_PATH"
                generate_rule
                apply_rule
                return
                ;;
            2)
                refresh_udev_rules
                return
                ;;
            *)
                print_warning "请输入 1 或 2"
                ;;
        esac
    done
}

main "$@"
