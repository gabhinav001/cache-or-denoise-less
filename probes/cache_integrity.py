"""CPU snapshots outside compiled regions detect cache mutation across calls."""
import torch


class CacheIntegrityMonitor:
    def __init__(self):
        self.entries = {}

    def watch(self, name, cache, keys):
        keys = set(keys)
        if not keys or not keys.issubset(cache) or name in self.entries:
            raise ValueError('Need a new name and complete nonempty cache coverage')
        snapshot = {k: cache[k].detach().to('cpu', copy=True) for k in keys}
        self.entries[name] = dict(cache=cache, snapshot=snapshot, checks=0,
                                  unchanged=True,
                                  finite=all(bool(torch.isfinite(v).all()) for v in snapshot.values()))

    def check(self):
        for entry in self.entries.values():
            entry['checks'] += 1
            for key, saved in entry['snapshot'].items():
                current = entry['cache'].get(key)
                if current is None:
                    entry['unchanged'] = False
                    continue
                current = current.detach().cpu()
                entry['finite'] &= bool(torch.isfinite(current).all())
                entry['unchanged'] &= (current.dtype == saved.dtype and current.shape == saved.shape
                                        and torch.equal(current.view(torch.uint8), saved.view(torch.uint8)))

    def report(self):
        return {name: dict(entries=len(e['snapshot']), checks=e['checks'],
                           unchanged=e['unchanged'], finite=e['finite'])
                for name, e in self.entries.items()}

    def passed(self):
        return bool(self.entries) and all(e['checks'] > 0 and e['unchanged'] and e['finite']
                                          for e in self.entries.values())
