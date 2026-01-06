{ config, pkgs, lib, ... }:

{
  # Create systemd service for ydotoold
  systemd.services.ydotoold = {
    description = "ydotool daemon";
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      ExecStart = "${pkgs.ydotool}/bin/ydotoold --socket-path=/run/ydotoold/socket --socket-perm=0660";
      Restart = "always";
      RuntimeDirectory = "ydotoold";
      RuntimeDirectoryMode = "0750";
      # Run as root to access /dev/uinput
      User = "root";
      Group = "input";
    };
  };

  # Add your user to input group for device access
  users.users.jordan.extraGroups = [ "input" ];
  
  # Create udev rule to make /dev/uinput accessible
  services.udev.extraRules = ''
    KERNEL=="uinput", GROUP="input", MODE="0660"
  '';
  
  # Create a wrapper script that uses the system socket
  environment.systemPackages = [
    (pkgs.writeScriptBin "ydotool-client" ''
      #!${pkgs.bash}/bin/bash
      exec ${pkgs.ydotool}/bin/ydotool --socket-path=/run/ydotoold/socket "$@"
    '')
  ];
}
