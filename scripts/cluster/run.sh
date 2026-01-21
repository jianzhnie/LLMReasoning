#!/bin/bash
# 自动配置root用户 ssh 免密登录
bash auto_ssh_config.sh \
    --file node_list_all.txt \
    --user root \
    --password Huawei12#$

# 为每个节点添加 user 用户
bash auto_add_user.sh \
    --file node_list_all.txt \
    --user fdd \
    --password pcl@0312 \
    --sudo

# 自动配置 user 用户 ssh 免密登录
bash auto_ssh_config.sh \
    --file node_list_all.txt \
    --user fdd \
    --password pcl@0312 \

# 自动构建 NFS 脚本
bash auto_build_nfs.sh \
    --server-ip 10.42.24.194 \
    --client-list node_list_all.txt \
    --share-path /home/fdd/workspace \
    --mount-point /home/fdd/workspace