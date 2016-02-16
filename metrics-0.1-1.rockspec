package = "metrics"
version = "0.1-1"
source = {
  url = "git://github.com/joeyhng/torch_metrics/"
}
dependencies = {
  "lua"
}
build = {
  type = "builtin",
  modules = {
    ['metrics'] = 'init.lua'
  }
}
