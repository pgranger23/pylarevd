#!/bin/bash
# Build the geometry shim inside the SL7 container, from the repo root:
#   ./inlar.sh "bash shim/build_shim.sh"
set -u
set -e
cd "$(dirname "$0")"
INCS="-I$GALLERY_INC -I$CANVAS_INC -I$CETLIB_INC -I$CETLIB_EXCEPT_INC -I$FHICLCPP_INC \
 -I$MESSAGEFACILITY_INC -I$HEP_CONCURRENCY_INC -I$LARDATAOBJ_INC -I$LARCOREOBJ_INC \
 -I$LARCOREALG_INC -I$LARDATAALG_INC -I$DUNECORE_INC -I$NUSIMDATA_INC -I$CANVAS_ROOT_IO_INC \
 -I$CLHEP_INC -I$BOOST_INC -I$ROOT_INC -I$RANGE_INC -I$TBB_INC -I$NLOHMANN_JSON_INC"
LIBS="-L$CETLIB_LIB -lcetlib -L$CETLIB_EXCEPT_LIB -lcetlib_except \
 -L$FHICLCPP_LIB -lfhiclcpp -lfhiclcpp_types \
 -L$MESSAGEFACILITY_LIB -lMF_MessageLogger \
 -L$LARCOREALG_LIB -llarcorealg_Geometry -L$LARDATAALG_LIB -llardataalg_DetectorInfo \
 -L$DUNECORE_LIB -ldunecore_Geometry \
 $(root-config --libs) -lGeom -lEG"
g++ -std=c++17 -O2 -fPIC -shared -o libevdgeom.so geom_shim.cc $INCS $LIBS
echo "built: $(pwd)/libevdgeom.so"
