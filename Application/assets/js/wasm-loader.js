// WASM 目录解析。
//
// 页面既可能在 OptimalTransport/ 下用本地静态服务打开，也可能被别的页面
// 以不同层级引用，所以按“当前页面所在目录”推导，避免写死相对层级。
(function () {
  var here = location.pathname.replace(/[^/]*$/, "");
  window.WasmLoader = {
    BASE: here + "assets/wasm/",
    url: function (name) {
      return this.BASE + name;
    }
  };
})();
