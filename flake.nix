{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        system = "${system}";
        config = {
          allowUnfree = true;
          permittedInsecurePackages = [
            "python3.11-gradio-3.44.3"
          ];
        };
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ 
          pkgs.python311
          pkgs.python311Packages.numpy
          pkgs.python311Packages.opencv4
          pkgs.python311Packages.gradio
          pkgs.python311Packages.torchvision-bin
        ];
      };
    };
}
