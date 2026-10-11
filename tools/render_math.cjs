// Render formulas at build time. Reading the output needs no math CDN or JS.
const fs = require("fs");
const path = require("path");
const katex = require(path.resolve(process.argv[2]));
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const results = input.map((item) => {
  try {
    return {
      html: katex.renderToString(item.tex, {
        displayMode: item.display, throwOnError: true,
        strict: "ignore", trust: false, output: "htmlAndMathml",
      }),
      error: null,
    };
  } catch (error) {
    return {html: "", error: String(error)};
  }
});
process.stdout.write(JSON.stringify(results));
