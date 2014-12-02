void kernel range_op(global float* A, const int size)
{
    const unsigned range = size / get_global_size(0);
    const unsigned start = get_global_id(0) * range;
    const unsigned end   = get_global_id(0) == get_global_size(0) ?
                           size : start + range;
    for (int i = start; i < end; ++i) {
        A[i] = sqrt(A[i]);
    }
}

void kernel element_op(global float* A)
{
    const unsigned pos = get_global_id(0);
    A[pos] = sqrt(A[pos]);
}

