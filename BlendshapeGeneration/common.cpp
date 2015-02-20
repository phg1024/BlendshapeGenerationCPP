#include "common.h"

#include "cholmod.h"

namespace global {
    static cholmod_common common;
    cholmod_common *cm;

    void initialize() {
      cout << "initializing global variables..." << endl;
      cholmod_start(&common);
      cm = &common;
      cout << "global variables initialized..." << endl;
    }

    void finalize() {
        cholmod_finish(&common);
    }
}
