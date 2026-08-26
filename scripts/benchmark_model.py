import torch
import argparse
import timeit
import statistics

from llm_systems.nn.transformer_lm import TransformerLM
from llm_systems.training.optimizer import AdamW
from llm_systems.training.loss import cross_entropy_loss

# command-line arguments

def parse_args():
    parser = argparse.ArgumentParser(
        description= 'Benchmark Custom Transformer Model'
    )

    # Data
    parser.add_argument(
        '--d-model',
        type= int,
        default= 768
    )

    parser.add_argument(
        '--d-ff',
        type= int,
        default= 3072
    )

    parser.add_argument(
        '--num-layers',
        type= int,
        default= 12
    )
    
    parser.add_argument(
        '--num-heads',
        type= int,
        default= 12
    )

    parser.add_argument(
        '--vocab-size',
        type= int,
        default= 10000
    )

    parser.add_argument(
        '--context-length',
        type= int,
        default= 512
    )

    parser.add_argument(
        '--warmup-steps',
        type= int,
        default= 10
    )

    parser.add_argument(
        '--measurement-steps',
        type= int,
        default= 100
    )

    parser.add_argument(
        '--batch-size',
        type= int,
        default= 4
    )

    parser.add_argument(
        '--rope-theta',
        type= float,
        default= 10000.0
    )

    parser.add_argument(
        '--device',
        type= str,
        default= 'auto'
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # setup-device
    if args.device == 'auto':
        device = (
            'cuda' if 
            torch.cuda.is_available()
            else 'cpu'
        )
    else:
        device = args.device

    print(f"Using device: {device}")

    # construct model
    model_dtype = torch.float32

    model = TransformerLM(
        vocab_size= args.vocab_size,
        context_length= args.context_length,
        num_layers= args.num_layers,
        d_model= args.d_model,
        num_heads= args.num_heads,
        d_ff= args.d_ff,
        theta= args.rope_theta,
        device= device,
        dtype= model_dtype
    )

    model.train()


    # Construct random token_ids
    inputs = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(
            args.batch_size,
            args.context_length,
        ),
        device=device,
    )

    targets = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(
            args.batch_size,
            args.context_length,
        ),
        device=device,
    )

    # optimizer
    optimizer = AdamW(
        params= model.parameters(),
        lr = 3e-4
    )

    # warmup loop
    for _ in range(args.warmup_steps):
        optimizer.zero_grad()

        logits = model(inputs)

        loss = cross_entropy_loss(
            logits,
            targets
        )

        loss.backward()

        del logits
        del loss


    # forward pass

    iteration_time = []


    for step in range(args.measurement_steps):
        
        # zero grad
        optimizer.zero_grad()

        # measurement
        # start time
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = timeit.default_timer()

        # forward
        logits = model(inputs)

        # loss 
        loss = cross_entropy_loss(
            logits,
            targets
        )

        # calculate backward
        loss.backward()

        # end time
        if device == "cuda":
            torch.cuda.synchronize()

        end_time = timeit.default_timer()

        # save time
        iteration_time.append((end_time - start_time))

        del logits
        del loss


    avg_time = sum(iteration_time) / len(iteration_time)
    std_time = statistics.stdev(iteration_time) if len(iteration_time) > 1 else 0.0

    avg_ms = avg_time * 1000
    std_ms = std_time * 1000

    print(
        f"Average forward + backward time: {avg_ms:.3f} ms\n"
        f"Standard deviation: {std_ms:.3f} ms"
    )
            
if __name__ == '__main__':
    main()