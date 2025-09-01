{
  description = "motherbrain dev environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }: let
    pkgs = import nixpkgs { system = "x86_64-linux"; };
    py = pkgs.python312;

    # Needed for save state support (otherwise core info folder is immutable, which causes issues)
    retroWithFceumm = pkgs.retroarch.withCores (cores: with cores; [ fceumm ]);
    fceummRunner = pkgs.writeShellScriptBin "fceumm" '' set -euo pipefail
      CORE="${pkgs.libretro.fceumm}/lib/libretro/fceumm_libretro.so"

      CFG="$PWD/.retroarch.cfg"
      BASE="$PWD/.retro"
      mkdir -p "$BASE"/{states,saves,system}

      cat > "$CFG" <<EOF
      savestate_directory = "$BASE/states"
      savefile_directory  = "$BASE/saves"
      system_directory    = "$BASE/system"
      core_info_directory = "${pkgs.libretro-core-info}/share/libretro/info"
EOF

      exec ${retroWithFceumm}/bin/retroarch -L "$CORE" --config "$CFG" "$@"
    '';

  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
      buildInputs = [
        py
        py.pkgs.venvShellHook

        pkgs.zlib

        # for generating the initial save state / interactive play
        fceummRunner
        pkgs.libretro-core-info
        pkgs.xorg.libX11
        pkgs.libGL
        pkgs.mesa
        pkgs.libGLU
      ];
      
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
        pkgs.zlib
        pkgs.gcc.cc.lib
        pkgs.xorg.libX11
        pkgs.libGL
        pkgs.mesa
        pkgs.libGLU
        # pkgs.SDL2
      ];

      venvDir = ".venv";
      postVenvCreation = ''
        pip install --upgrade pip
        pip install stable-retro
        pip install -r dreamerv3/requirements.txt
      '';
      postShellHook = ''
        export RETRO_DATA_PATH=$PWD/data
        echo "Python environment ready"
        echo $RETRO_DATA_PATH
        ls $RETRO_DATA_PATH/stable/Metroid-Nes
        cat $RETRO_DATA_PATH/stable/Metroid-Nes/rom.sha
      '';
    };
  };
}

