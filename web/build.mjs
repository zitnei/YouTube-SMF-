// ブラウザ版をビルドして ../docs に出力する（GitHub Pages はこのフォルダを公開する）
import { build } from "esbuild";
import { fileURLToPath } from "url";
import { cpSync, mkdirSync, rmSync, writeFileSync } from "fs";

const out = new URL("../docs/", import.meta.url);
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

await build({
  entryPoints: ["src/main.js"],
  bundle: true,
  minify: true,
  format: "esm",
  target: "es2020",
  platform: "browser",
  outfile: fileURLToPath(new URL("app.js", out)),
  legalComments: "linked",
  logLevel: "info",
});

cpSync("src/index.html", new URL("index.html", out));
cpSync("node_modules/@spotify/basic-pitch/model", new URL("model", out), { recursive: true });
writeFileSync(new URL(".nojekyll", out), ""); // GitHub Pages の加工を止める
console.log("built -> docs/");
