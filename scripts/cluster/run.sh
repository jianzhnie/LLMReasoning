#!/bin/bash
# 自动配置root用户 ssh 免密登录
bash auto_config_ssh.sh \
    --file node_list.txt \
    --user root \
    --password 123456

# 为每个节点添加 user 用户
bash auto_add_user.sh \
    --file node_list.txt \
    --user jianzhnie \
    --password 123456 \
    --sudo

# 自动配置 user 用户 ssh 免密登录
bash auto_config_ssh.sh \
    --file node_list.txt \
    --user jianzhnie \
    --password 123456 \

# 自动构建 NFS 脚本
bash auto_build_nfs.sh \
    --node-list node_list_all.txt \
    --share-path /home/jianzhnie/llmtuner \
    --mount-point /home/jianzhnie/llmtuner


# 自动构建 NFS 脚本
bash auto_build_nfs.sh \
    --node-list node_list_all.txt \
    --share-path /home/fdd/workspace \
    --mount-point /home/fdd/workspace
