# Vagrant.configure("2") do |config|
#     # It would be nice to use bionic64 (Ubuntu 18.04 LTS), but
#     # tcp_probe is broken on the corresponding kernel, 4.15
#     # (replacement was added in 4.16).
#     # So, use 17.10 with kernel 4.13.
#   config.vm.box = "generic/ubuntu1710"
#   config.vm.post_up_message = ""\
#     "Welcome to CCP. "\
#     "Run `make` in /ccp to compile. This may take some time. "
#   config.vm.synced_folder ".", "/park",
#     id: "park"
#   config.vm.provision "shell", path: "./park/envs/congestion_control/ccp-system-setup.sh"
# end
Vagrant.configure("2") do |config|
  config.vm.box = "perk/ubuntu-2204-arm64"
  config.vm.provider "qemu" do |qe|
    qe.ssh_port = "50022" # change ssh port as needed
  end
end