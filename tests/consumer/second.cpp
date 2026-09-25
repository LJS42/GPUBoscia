#include <gpuboscia/problem.h>
#include <gpuboscia/branching.h>
#include <gpuboscia/gap_utils.h>
#include <gpuboscia/line_search_options.h>
#include <gpuboscia/cpu_relax.h>

bool second_translation_unit() {
    return gpuboscia::gap_absolute(1.0, 3.0) == 2.0;
}
