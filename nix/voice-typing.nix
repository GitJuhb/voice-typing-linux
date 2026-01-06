{ config, lib, pkgs, ... }:

let
  cfg = config.services.voiceTyping;
in
{
  options.services.voiceTyping = {
    enable = lib.mkEnableOption "voice typing user service";

    command = lib.mkOption {
      type = lib.types.str;
      default = "";
      example = "/home/jordan/voice-typing-nix/voice";
      description = "Command to start voice typing (use an absolute path).";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = {};
      example = { VOICE_LOG_FILE = "/home/jordan/.local/state/voice-typing/voice-typing.log"; };
      description = "Environment variables passed to the service.";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.command != "";
        message = "services.voiceTyping.command must be set.";
      }
    ];

    systemd.user.services.voice-typing = {
      Unit = {
        Description = "Voice Typing";
      };
      Service = {
        ExecStart = cfg.command;
        Restart = "on-failure";
        Environment = lib.mapAttrsToList (n: v: "${n}=${v}") cfg.environment;
      };
      Install = {
        WantedBy = [ "default.target" ];
      };
    };
  };
}
