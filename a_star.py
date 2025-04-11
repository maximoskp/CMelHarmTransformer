import heapq
import torch
import torch.nn.functional as F

class SearchNode:
    def __init__(self, tokens, logprob, heuristic, parent=None):
        self.tokens = tokens  # shape: (1, seq_len)
        self.logprob = logprob
        self.heuristic = heuristic
        self.parent = parent
        self.tried_tokens = set()  # token IDs already expanded

    def __lt__(self, other):
        return (self.logprob + self.heuristic) > (other.logprob + other.heuristic)
# end class SearchNode

def consistency_checker(tokens):
    consistent = True
    current_bar_time = None
    for i in range( 1, len(tokens), 1 ):
        # no melody-related tokens here
        if 'P:' in tokens[i] or 'rest' in tokens[i] or \
            'fill' in tokens[i] or '<s>' in tokens[i] \
            or 'ts_' in tokens[i]:
            consistent = False
            break
        # no two consequtive position tokens
        if 'position_' in tokens[i] and 'position_' in tokens[i-1]:
            consistent = False
            break
        # only chord / pc after position
        if 'position_' in tokens[i-1] and (
            'bar' in tokens[i] or '</s>' in tokens[i] or 
            '<h>' in tokens[i] or '</m>' in tokens[i]
        ):
            consistent = False
            break
        # only position, bar, </m>, </s> and <h> after bar
        if 'bar' in tokens[i-1]:
            if 'position_' in tokens[i]:
                current_bar_time = float( tokens[i].split('_')[-1].replace('x', '.') )
            elif 'bar' not in tokens[i] and '</m>' not in tokens[i] and \
                '</s>' not in tokens[i] and '<h>' not in tokens[i]:
                consistent = False
                break
        # time should increase within bar
        if 'position_' in tokens[i] and 'bar' not in tokens[i-1]:
            if current_bar_time is None:
                consistent = False
                break
            if float( tokens[i].split('_')[-1].replace('x', '.') ) <= current_bar_time:
                consistent = False
                break
            else:
                current_bar_time = float( tokens[i].split('_')[-1].replace('x', '.') )
    return consistent
    # end consistency_checker

class AStarGPT:
    def __init__(self, model, tokenizer, input_ids, constraint_ids, max_length=512, beam_width=10, lookahead_k=5):
        self.model = model
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.constraint_ids = constraint_ids.tolist()
        self.max_length = max_length
        self.beam_width = beam_width
        self.lookahead_k = lookahead_k
        self.eos_token_id = tokenizer.eos_token_id
        self.eos_token = tokenizer.eos_token
        self.constraint_tokens_breakdown()
    # end init

    def constraint_tokens_breakdown(self):
        tokens = self.tokenizer.convert_ids_to_tokens(self.constraint_ids)
        bar_count = 0
        chord_tokens = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if 'bar' in tok or 'fill' in tok or '/m' in tok or '<h>' in tok:
                if 'bar' in tok:
                    bar_count += 1
                i += 1
            else:
                # we should have arrived in a position token
                position_token = tokens[i]
                # the remaining tokens should be the chord
                i += 1
                while i < len(tokens) and \
                    'fill' not in tokens[i] and \
                     'bar' not in tokens[i] and \
                     '</s>' not in tokens[i]:
                    chord_tokens.append( tokens[i] )
                    i += 1
                break
        self.constraint_bar = bar_count
        self.position_token = position_token
        self.chord_tokens = chord_tokens
    # end constraint_tokens_breakdown

    def constraint_checker(self, input_tokens):
        tokens = self.tokenizer.convert_ids_to_tokens(input_tokens.tolist())
        start_harmonization_index = tokens.index('<h>')
        tokens = tokens[start_harmonization_index:]
        # print(f'constraint_checker: {tokens}')
        # print(f'num_bars: {tokens.count('<bar>')} - num_tokens: {len(tokens)}')
        if not consistency_checker(tokens):
            # print('inconsistent')
            return False
        
        bar_count = 0
        found = False
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "<bar>":
                bar_count += 1
            elif bar_count == self.constraint_bar and tok == self.position_token:
                j = 0
                found = True
                while j < len(self.chord_tokens):
                    if i + j + 1 < len(tokens):
                        if tokens[i + j + 1] != self.chord_tokens[j]:
                            found = False
                            break
                    else:
                        break
                    j += 1
                # break
                # no break, we need to keep counting bars to check premature ending
            i += 1
        # if the sequence has reached eos and the constraint has not been met, it should fail
        if tokens[-1] == self.eos_token:
            condition = found
            # print(f'checker EOS: {condition}')
        elif len(tokens) == self.max_length:
            condition = False
            # print(f'checker PRE: {condition}')
        else:
            condition = found or bar_count <= self.constraint_bar  # Only reject if we're past bar of interest and it's missing
            # print(f'checker PRE: {condition}')
        return condition
    # end constraint_checker

    def expand_node(self, node):
        with torch.no_grad():
            output = self.model(node.tokens.to(self.model.device), return_dict=True)
            logits = output.logits[:, -1, :]

            # Mask out already visited tokens before softmax
            if node.tried_tokens:
                mask = torch.full_like(logits, 0.0)
                mask[:, list(node.tried_tokens)] = float('-inf')
                logits = logits + mask

            probs = F.log_softmax(logits, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, self.lookahead_k, dim=-1)

        new_nodes = []
        for i in range(self.lookahead_k):
            token_id = topk_ids[0, i].item()
            token_prob = topk_probs[0, i].item()
            node.tried_tokens.add(token_id)

            new_tokens = torch.cat(
                [node.tokens.to(self.model.device), topk_ids[0, i].unsqueeze(0).unsqueeze(0)], dim=-1
            ).to(self.model.device)

            if not self.constraint_checker(new_tokens[0]):
                continue
            
            new_logprob = node.logprob + token_prob
            # print('new_logprob:', new_logprob)
            new_node = SearchNode(new_tokens, new_logprob, 0.0, parent=node)
            new_nodes.append(new_node)
        return new_nodes
    # end expand_node

    def decode(self):
        initial_node = SearchNode(tokens=self.input_ids, logprob=0.0, heuristic=0.0)
        open_set = [initial_node]
        finished = []
        while open_set:
            current = heapq.heappop(open_set)

            # if current.tokens.shape[-1] >= self.max_length or (
            #     self.eos_token_id and current.tokens[0, -1].item() == self.eos_token_id
            # ):
            #     finished.append(current)
            #     continue
            if current.tokens[0, -1].item() == self.eos_token_id and \
                self.constraint_checker(current.tokens[0]):
                finished.append(current)
                continue
            if current.tokens.shape[-1] >= self.max_length:
                continue

            # Expand current node
            # print('children')
            children = self.expand_node(current)

            if children:
                for child in children:
                    heapq.heappush(open_set, child)
            else:
                # No valid expansions – backtrack to unvisited options
                back = current.parent
                while back:
                    # Re-expand from back with unvisited options
                    # print('parents')
                    more_options = self.expand_node(back)
                    if more_options:
                        for opt in more_options:
                            heapq.heappush(open_set, opt)
                        break
                    back = back.parent

            # Prune open set
            open_set = sorted(open_set, reverse=False)[:self.beam_width]
            # print('finished:', len(finished))
            # just keep the first one found
            if len(finished) >= 1:
                break

        if not finished:
            raise RuntimeError("No valid sequence could be generated under constraints.")

        best = sorted(finished, key=lambda x: x.logprob + x.heuristic, reverse=True)[0]
        return best.tokens, finished
    # end decode
# end class AStar

class AStarBART:
    def __init__(self, model, tokenizer, encoder_input_ids, constraint_ids, max_length=128, beam_width=10, lookahead_k=5):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder_input_ids = encoder_input_ids.to(model.device)
        self.constraint_ids = constraint_ids.tolist()
        self.max_length = max_length
        self.beam_width = beam_width
        self.lookahead_k = lookahead_k
        self.eos_token_id = tokenizer.eos_token_id
        self.eos_token = tokenizer.eos_token
        self.constraint_tokens_breakdown()

        with torch.no_grad():
            self.encoder_outputs = model.model.encoder(input_ids=self.encoder_input_ids, return_dict=True)
    # end init

    def constraint_tokens_breakdown(self):
        tokens = self.tokenizer.convert_ids_to_tokens(self.constraint_ids)
        bar_count = 0
        chord_tokens = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if 'bar' in tok or 'fill' in tok or '/m' in tok or '<h>' in tok:
                if 'bar' in tok:
                    bar_count += 1
                i += 1
            else:
                # we should have arrived in a position token
                position_token = tokens[i]
                # the remaining tokens should be the chord
                i += 1
                while i < len(tokens) and \
                    'fill' not in tokens[i] and \
                     'bar' not in tokens[i] and \
                     '</s>' not in tokens[i]:
                    chord_tokens.append( tokens[i] )
                    i += 1
                break
        self.constraint_bar = bar_count
        self.position_token = position_token
        self.chord_tokens = chord_tokens
    # end constraint_tokens_breakdown

    def constraint_checker(self, input_tokens):
        tokens = self.tokenizer.convert_ids_to_tokens(input_tokens.tolist())
        # print(f'constraint_checker: {tokens}')
        # print(f'num_bars: {tokens.count('<bar>')} - num_tokens: {len(tokens)}')
        if not consistency_checker(tokens):
            # print('inconsistent')
            return False
        
        bar_count = 0
        found = False
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "<bar>":
                bar_count += 1
            elif bar_count == self.constraint_bar and tok == self.position_token:
                j = 0
                found = True
                while j < len(self.chord_tokens):
                    if i + j + 1 < len(tokens):
                        if tokens[i + j + 1] != self.chord_tokens[j]:
                            found = False
                            break
                    else:
                        break
                    j += 1
                # break
                # no break, we need to keep counting bars to check premature ending
            i += 1
        # if the sequence has reached eos and the constraint has not been met, it should fail
        if tokens[-1] == self.eos_token:
            condition = found
            # print(f'checker EOS: {condition}')
        elif len(tokens) == self.max_length:
            condition = False
            # print(f'checker PRE: {condition}')
        else:
            condition = found or bar_count <= self.constraint_bar  # Only reject if we're past bar of interest and it's missing
            # print(f'checker PRE: {condition}')
        return condition
    # end constraint_checker

    def expand_node(self, node):
        decoder_input_ids = node.tokens.to(self.model.device)

        with torch.no_grad():
            output = self.model.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=self.encoder_outputs.last_hidden_state,
                return_dict=True
            )
            logits = self.model.lm_head(output.last_hidden_state[:, -1, :])
            if node.tried_tokens:
                mask = torch.full_like(logits, 0.0)
                mask[:, list(node.tried_tokens)] = float('-inf')
                logits = logits + mask

            probs = F.log_softmax(logits, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, self.lookahead_k, dim=-1)

        new_nodes = []
        for i in range(self.lookahead_k):
            token_id = topk_ids[0, i].item()
            token_prob = topk_probs[0, i].item()
            node.tried_tokens.add(token_id)

            new_tokens = torch.cat([decoder_input_ids, topk_ids[0, i].unsqueeze(0).unsqueeze(0)], dim=-1)

            if not self.constraint_checker(new_tokens[0]):
                continue
            
            new_logprob = node.logprob + token_prob
            new_node = SearchNode(new_tokens, new_logprob, 0.0, parent=node)
            new_nodes.append(new_node)

        return new_nodes
    # end expand_node

    def decode(self):
        initial_node = SearchNode(tokens=torch.tensor([[self.tokenizer.vocab['<s>']]], device=self.model.device), logprob=0.0, heuristic=0.0)
        open_set = [initial_node]
        finished = []

        while open_set:
            current = heapq.heappop(open_set)
            if current.tokens.shape[-1] >= self.max_length:
                continue
            if current.tokens[0, -1].item() == self.eos_token_id and self.constraint_checker(current.tokens[0]):
                finished.append(current)
                continue

            children = self.expand_node(current)
            if children:
                for child in children:
                    heapq.heappush(open_set, child)
            else:
                back = current.parent
                while back:
                    more_options = self.expand_node(back)
                    if more_options:
                        for opt in more_options:
                            heapq.heappush(open_set, opt)
                        break
                    back = back.parent

            open_set = sorted(open_set, reverse=False)[:self.beam_width]
            if len(finished) >= 1:
                break

        if not finished:
            raise RuntimeError("No valid sequence could be generated under constraints.")

        best = sorted(finished, key=lambda x: x.logprob + x.heuristic, reverse=True)[0]
        return best.tokens, finished
    # end decode
# end class AStarBart