// A second translation unit catches non-inline definitions in installed headers.
#include <gpuboscia/gpuboscia.cuh>

gpuboscia::Settings consumer_default_settings() {
    return gpuboscia::create_default_settings();
}
