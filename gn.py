from pyspark import SparkContext
import os, argparse, time, copy
import numpy as np
from collections import namedtuple, defaultdict
from queue import Queue

Edge = namedtuple("Edge", ["src", "dst"])

class GN(object):
	def __init__(self, num_partition):
		self._num_partition = num_partition
		self._const_linked_list = None
		self._degree = None
		self._m = None

	def readfile(self, spark_context, input_file_path):
		sc = spark_context
		RDD = sc.textFile(input_file_path) \
				.repartition(self._num_partition)
		return RDD

	def get_edges(self, RDD):
		RDD = RDD.flatMap(self.get_edge_tuple)
		return RDD

	def get_vertices(self, RDD):
		RDD = RDD.flatMap(self.get_split).distinct()
		return RDD

	def get_m(self, edges):
		self._m = edges.count()
		return self._m

	def get_linked_list(self, vertices, edges):
		linked_list = defaultdict(set)
		for v in vertices:
			for e in edges:
				if e.src == v:
					linked_list[v].add(e.dst)
				if e.dst == v:
					linked_list[v].add(e.src)
		self._const_linked_list = copy.deepcopy(linked_list)
		return linked_list

	def get_vertice_degree(self, linked_list):
		degree = defaultdict(int)
		for v in linked_list:
			degree[v] = len(linked_list[v])
		self._degree = degree
		return degree

	def update(self, edge, linked_list, communitys):
		u, v = edge
		linked_list[u].remove(v)
		linked_list[v].remove(u)
		flag, split = self.if_split(edge, linked_list)
		if flag:
			for community in communitys:
				if split.issubset(community):
					communitys.remove(community)
					communitys.add(split)
					communitys.add(community - split)
					Q = self.find_modularity(m=self._m, linked_list=self._const_linked_list, 
								communitys=communitys, degree=self._degree)
					return Q

	@staticmethod
	def if_split(edge, linked_list):
		u, v = edge
		queue = []
		queue.append(u)
		visited = set()
		visited.add(u)
		flag = True
		while queue:
			cur = queue.pop(0)
			if cur == v:
				flag = False
			for child in linked_list[cur]:
				if child not in visited:
					visited.add(child)
					queue.append(child)
		return flag, frozenset(visited)

	def find_modularity(self, m, linked_list, communitys, degree):
		expect_m = 0
		num_edge = 0
		for community in communitys:
			# print(community)
			for i in community:
				for j in community:
					expect_m += degree[i] * degree[j]
					if j in linked_list[i]:
						num_edge += 1
		# print("### of edges", num_edge)
		# print("expect of edges", expect_m)
		Q = 0.5 / m * (num_edge - expect_m * 0.5 / m)
		return Q

	def find_betweenness(self, vertice, linked_list):
		betweenness = vertice.flatMap(lambda v: self.bfs(vertice=v, linked_list=linked_list)) \
					.reduceByKey(lambda a,b: a+b) \
					.map(lambda x: [sorted(x[0]), x[1]/2])
		betweenness = betweenness.collect()
		betweenness = sorted(betweenness, key=lambda x: (-x[1], x[0]))
		return betweenness

	def bfs(self, vertice, linked_list):
		visited, node_credits = self.forward(vertice=vertice, linked_list=linked_list)
		edge_weights = self.backward(node_weights=visited, node_credits=node_credits, linked_list=linked_list)
		return edge_weights

	@staticmethod
	def backward(node_weights, node_credits, linked_list):
		edge_weights = []
		cur_row = node_credits.pop()
		while node_credits:
			next_row = node_credits.pop()
			for u in cur_row:
				for v in next_row:
					if v in linked_list[u]:
						ratio = node_weights[v] / node_weights[u]
						edge_weight = cur_row[u] * ratio
						next_row[v] += edge_weight
						edge_weights.append([frozenset([u,v]), edge_weight])
			cur_row = next_row
		return edge_weights

	@staticmethod
	def forward(vertice, linked_list):
		def bfs_row(row):
			next_row = []
			node_credits = defaultdict(int)
			row = copy.deepcopy(row)
			temp_visited = defaultdict(int)
			while row:
				cur = row.pop(0)
				for child in linked_list[cur]:
					if child not in visited:
						next_row.append(child)
						node_credits[child] = 1
			return next_row, node_credits

		def update_visited(visited, childs):
			for c in childs:
				visited[c] += 1
			return visited

		visited = defaultdict(int)
		visited[vertice] += 1
		check_queues = []
		check_queues.append([vertice])
		node_credits = []
		node_credits.append(copy.deepcopy(visited))
		while True:
			childs, credits = bfs_row(check_queues[-1])
			if childs:
				visited = update_visited(visited, childs)
				check_queues.append(childs)
				node_credits.append(credits)
			else:
				break
		return visited, node_credits

	@staticmethod
	def print_between(data, output_file_path):
		with open(output_file_path, 'w') as f:
			for i in data:
				f.write(str(i).replace("[[","(").replace("']","')").replace("]",""))
				f.write("\n")

	@staticmethod
	def print_community(data, output_file_path):
		with open(output_file_path, 'w') as f:
			data = [sorted(c) for c in data]
			data = sorted(data, key=lambda x: (len(x), x[0]))
			for i in data:
				f.write(str(i).replace("[","").replace("]",""))
				f.write("\n")

	@staticmethod
	def get_edge_tuple(x):
		x = sorted(x.split(' '), reverse=True)
		return [Edge(src=x[0], dst=x[1])]

	@staticmethod
	def get_split(x):
		x = x.split(' ')
		return [x[0], x[1]]

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("input", default="yelp_train.csv", help="input yelp file")
	parser.add_argument("between_output", default="output.csv", help="output file")
	parser.add_argument("community_output", default="output.csv", help="output file")
	args = parser.parse_args()

	sc = SparkContext('local[*]', 'FrequentItems')
	sc.setLogLevel("ERROR")

	start_time = time.time()
	### read files
	gn = GN(num_partition=5)
	rdd = gn.readfile(spark_context=sc, input_file_path=args.input)
	### data processing
	edges = gn.get_edges(rdd)
	vertices = gn.get_vertices(rdd)
	m = gn.get_m(edges)
	linked_list = gn.get_linked_list(vertices=vertices.collect(), edges=edges.collect())
	degree = gn.get_vertice_degree(linked_list=linked_list)
	communitys = set()
	communitys.add(frozenset(vertices.collect()))
	### print first betweenness
	betweenness = gn.find_betweenness(vertices, linked_list)
	gn.print_between(data=betweenness, output_file_path=args.between_output)
	### find best partition
	result = []
	while betweenness:
		Q = gn.update(edge=betweenness[0][0], linked_list=linked_list, 
					communitys=communitys)
		betweenness = gn.find_betweenness(vertices, linked_list)
		if Q:
			# print(Q)
			result.append([Q, copy.deepcopy(communitys)])
	result = sorted(result, reverse=True)
	# print(result[0][1])
	gn.print_community(data=result[0][1], output_file_path=args.community_output)

	total_time = time.time() - start_time
	print("Duration:", total_time)

if __name__ == '__main__':
	main()