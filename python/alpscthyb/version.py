import pkg_resources

try:
    version = pkg_resources.get_distribution('alpscthyb').version
except:
    version = ""

def show_version():
  print("\nYou are using the ALPS/CT-HYB version %s\n"%version)
