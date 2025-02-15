{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        mpkgs = import inputs.nixpkgs_master {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        libList =
          [
            # Add needed packages here
            pkgs.stdenv.cc.cc
            pkgs.libGL
            pkgs.glib
            pkgs.zlib
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.isLinux (
            with pkgs;
            [
             # cudatoolkit

              # This is required for most app that uses graphics api
              # linuxPackages.nvidia_x11
            ]
          );
      in
      with pkgs;
      {
        devShells = {
          default =
            let
              python_with_pkgs = pkgs.python311.withPackages (pp: [
                # Add python pkgs here that you need from nix repos
		ruff
              ]);
            in
            mkShell {
              NIX_LD = runCommand "ld.so" { } ''
                ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
              '';
              NIX_LD_LIBRARY_PATH = lib.makeLibraryPath libList;
              packages = [
                python_with_pkgs
                python311Packages.venvShellHook
                # We # We now recommend to use uv for package management inside nix env
                mpkgs.uv

                # duckdb
              ] ++ libList;
              venvDir = "./.venv";
              postVenvCreation = ''
                unset SOURCE_DATE_EPOCH
              '';
              postShellHook = ''
                unset SOURCE_DATE_EPOCH
              '';
              shellHook = ''
                export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
                export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
                runHook venvShellHook
                uv pip install -e .
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
              '';
            };
        };
      }
    );
}
# Things one might need for debugging or adding compatibility
# export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
# export LD_LIBRARY_PATH=${pkgs.cudaPackages.cuda_nvrtc}/lib
# export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
# export EXTRA_CCFLAGS="-I/usr/include"

# Data syncthing commands
# syncthing cli show system | jq .myID
# syncthing cli config devices add --device-id $DEVICE_ID_B
# syncthing cli config folders $FOLDER_ID devices add --device-id $DEVICE_ID_B
# syncthing cli config devices $DEVICE_ID_A auto-accept-folders set true
