package = "metrics"
version = "0.1-1"
source = {
  url = "file:///home/joe/code/torch_metrics/"
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
