module Training

using ..Utilities
export AbstractTFRecur, 
    TFRecur,
    WeakTFRecur,
    init_state!,
    force,
    train_!,
    mar_loss,
    AR_convergence_loss,
    sample_batch,
    sample_sequence,
    AbstractDataset,
    Dataset,
    regularize,
    Progress,
    update!,
    print_progress

include("dataset.jl")
include("forcing.jl")
include("tfrecur.jl")
include("regularization.jl")
include("progress.jl")
include("training_rnn.jl")

end
